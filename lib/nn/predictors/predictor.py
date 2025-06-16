from typing import Optional
from torchmetrics import MetricCollection
from tsl.engines import Predictor as TSLPredictor
import lib.metrics
from lib.metrics.sample_metrics import SampleMetric
from lib.nn.utils import maybe_cat_v


class Predictor(TSLPredictor):
    
    def forward(self, *args, **kwargs):
        """"""
        v = kwargs.pop("v", None)
        kwargs["u"] = maybe_cat_v(kwargs.pop("u"), v)
        return super(Predictor, self).forward(*args, **kwargs)


class SamplingPredictor(Predictor):
    def __init__(self, *args, 
                 loss_fn = None, 
                #  scale_target = False, metrics = None, 
                 mc_samples_eval: int = 11,
                 mc_samples_train: int = 16,
                 mc_samples_test: Optional[int] = None,
                 **kwargs):
        if loss_fn is None:
            loss_fn = lib.metrics.EnergyScore()
        super(SamplingPredictor, self).__init__(*args, loss_fn=loss_fn, **kwargs)

        self.mc_samples_train = mc_samples_train
        self.mc_samples_eval = mc_samples_eval
        self.mc_samples_test = mc_samples_eval if mc_samples_test is None else mc_samples_test
        self.in_testing_step = False

    @property
    def mc_samples(self):
        if self.training:
            return self.mc_samples_train
        elif self.in_testing_step:
            return self.mc_samples_test
        else:
            return self.mc_samples_eval

    def forward(self, *args, force_mc_samples: bool = False, **kwargs):
        """"""
        if force_mc_samples:
            mc_samples = kwargs.pop("mc_samples")
        else:
            mc_samples = self.mc_samples
        return super(SamplingPredictor, self).forward(*args, mc_samples=mc_samples, **kwargs)

    def _set_metrics(self, metrics):
        sample_metrics, point_metrics = {}, {}
        for k, m in metrics.items():
            if isinstance(m, SampleMetric):
                sample_metrics[k] = m
            else:
                point_metrics[k] = m
        super(SamplingPredictor, self)._set_metrics(metrics=sample_metrics)
        self.test_point_metrics = MetricCollection(
            metrics={k: self._check_metric(m)
                     for k, m in point_metrics.items()},
            prefix='test_')
        
    def test_step(self, batch, batch_idx):
        self.in_testing_step = True
        # test sampling metrics with possibly a different number of samples
        test_loss = super(SamplingPredictor, self).test_step(batch, batch_idx)
        
        # test point prediction metrics
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True, force_mc_samples=True, mc_samples=None)

        y, mask = batch.y, batch.get('mask')
        self.test_point_metrics.update(y_hat.detach(), y, mask)
        self.log_metrics(self.test_point_metrics, batch_size=batch.batch_size)

        self.in_testing_step = False
        return test_loss