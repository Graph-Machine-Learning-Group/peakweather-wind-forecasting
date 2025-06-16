from typing import Optional, Tuple
from einops import rearrange
import torch
from torch import Tensor
from torchmetrics.utilities.checks import _check_same_shape
from tsl.metrics.torch import MaskedMetric 


def _check_input_shape(preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """ 
    Here we assume samples are in the first dimension, i.e., 
        target: [batch, ...]
        preds: [num_samples, batch, ...]
    """
    assert preds.ndim == target.ndim + 1
    _check_same_shape(preds[0], target)
    return preds, target 


class SampleMetric(MaskedMetric):
    """
    Extends `tsl.metrics.torch.MaskedMetric` by assuming `y_hat` is represented
    as a set of samples to model p(y). It is required that `y_hat` has the
    number of samples as leading dimension: y_hat.shape = [samples, *y.shape]. 
    """
    full_state_update: bool = False

    def __init__(self, metric_fn, mask_nans=False, mask_inf=False, metric_fn_kwargs=None, at=None, full_state_update = None, **kwargs):
        super().__init__(metric_fn, mask_nans, mask_inf, metric_fn_kwargs, at, full_state_update, **kwargs)
        self.eval_at = at is not None 
        self.sample_dim = 0  # used to rely on MaskedMetric evaluation at specific time steps

    def update(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        if self.eval_at:
            y_hat = rearrange(y_hat, "s b t ... -> b t s ...")
            self.sample_dim = 2
        super(SampleMetric, self).update(y_hat, y, mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, int]:
        if self.eval_at:
            y_hat = rearrange(y_hat, "b t s ... -> s b t ...")
            self.sample_dim = 0

        y_hat, y = _check_input_shape(preds=y_hat, target=y)
        val = self.metric_fn(y_hat, y)
        return val.sum(), val.numel()
    
    def _compute_masked(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.eval_at:
            y_hat = rearrange(y_hat, "b t s ... -> s b t ...")
            self.sample_dim = 0

        y_hat, y = _check_input_shape(preds=y_hat, target=y)
        val = self.metric_fn(y_hat, y)
        mask = self._check_mask(mask, val)
        val = torch.where(mask, val, torch.zeros_like(val))
        return val.sum(), mask.sum()