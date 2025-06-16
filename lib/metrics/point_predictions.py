from lib.metrics.sample_metrics import SampleMetric
from lib.metrics.functional.sampling import sampling_MAE, sampling_MSE


class SampleMAE(SampleMetric):
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(self, **kwargs):
        super(SampleMAE, self).__init__(metric_fn=sampling_MAE, metric_fn_kwargs=kwargs.pop("metric_fn_kwargs", dict()), **kwargs)


class SampleMSE(SampleMetric):
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(self, **kwargs):
        super(SampleMSE, self).__init__(metric_fn=sampling_MSE, metric_fn_kwargs=kwargs.pop("metric_fn_kwargs", dict()), **kwargs)
        