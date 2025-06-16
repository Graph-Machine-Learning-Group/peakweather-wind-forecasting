from typing import Optional
import torch

from torchmetrics.utilities.checks import _check_same_shape

from lib.metrics.functional import energy_score
from lib.metrics.sample_metrics import SampleMetric


class EnergyScore(SampleMetric): 
    
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(self, *args, **kwargs):
        super(EnergyScore, self).__init__(metric_fn=energy_score, *args, **kwargs)

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(val, dtype=torch.bool)
        else:
            mask = mask.bool()
            assert mask.shape[:-1] == val.shape
            mask = mask.all(-1)
            _check_same_shape(mask, val)
        if self.mask_nans:
            mask = mask & ~torch.isnan(val)
        if self.mask_inf:
            mask = mask & ~torch.isinf(val)
        return mask