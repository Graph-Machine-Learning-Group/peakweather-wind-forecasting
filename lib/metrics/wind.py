from typing import Callable, Optional

import torch
from torch.nn import functional as torch_functional 
from lib.metrics.crps import EnergyScore
from lib.metrics.functional.sampling import sampling_cosine_similarity

from tsl.metrics import torch_metrics

from lib.metrics.point_predictions import SampleMAE
from lib.metrics.sample_metrics import SampleMetric


def _mask_absent_wind(target: torch.Tensor, mask: Optional[torch.Tensor] = None, min_norm: float = 1e-3) -> torch.Tensor:
    """Utility to create a mask for nearly absent wind. 
    This is useful when computing metrics related to wind direction, 
    as low wind speeds can lead to unreliable directional data.
    The masking is computed with respect to the target wind.

    Args    :
        target (torch.Tensor): The target wind (u,v components).
        mask (torch.Tensor, optional): A pre-existing mask. Defaults to None.
        min_norm (float, optional): The minimum vector norm (i.e., wind speed) required for 
            wind to be considered valid. 
            Default to 1e-3.

    Returns:
        torch.Tensor: A new mask considering the optionally given one, masking away time steps
            where the wind is too soft.
    """  

    if min_norm is None:
        min_norm = -1
    wind_mask = torch.norm(target, dim=-1) > min_norm
    wind_mask = wind_mask.unsqueeze(-1)
    if mask is not None:
        mask = mask.bool()
        wind_mask = torch.logical_and(mask, wind_mask)
    return wind_mask


def _collapse_mask(mask: Optional[torch.Tensor], val: torch.Tensor):
    if mask is not None:
        assert mask.shape[:-1] == val.shape
        mask = mask.bool().all(-1)
    return mask


def _compute_angle(y_hat: torch.Tensor, y: torch.Tensor, cosine_similarity_fun: Callable):
    cos_phi = cosine_similarity_fun(y_hat, y, dim=-1)
    cos_phi = torch.clamp(cos_phi, -1.0, 1.0)
    phi = torch.acos(cos_phi)
    phi = torch.rad2deg(phi)
    assert (phi >= 0).all() and (phi<=180).all()
    return phi
    

def sampling_dir_mae(y_hat: torch.Tensor, y: torch.Tensor):
    return _compute_angle(y_hat, y, sampling_cosine_similarity)


class SpeedMAE(torch_metrics.MaskedMAE): 
    """ MAE on wind speed computed as the norm of wind velocity. """

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(SpeedMAE, self)._check_mask(mask, val)

    def _compute_masked(self, y_hat: torch.Tensor, y, mask: Optional[torch.Tensor]):
        return super(SpeedMAE, self)._compute_masked(torch.norm(y_hat, dim=-1), torch.norm(y, dim=-1), mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return super(SpeedMAE, self)._compute_std(torch.norm(y_hat, dim=-1), torch.norm(y, dim=-1))


class SampleSpeedMAE(SampleMAE): 
    """ MAE on wind speed based of predicted samples. """

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(SampleSpeedMAE, self)._check_mask(mask, val)

    def _compute_masked(self, y_hat: torch.Tensor, y, mask: Optional[torch.Tensor]):
        return super(SampleSpeedMAE, self)._compute_masked(torch.norm(y_hat, dim=-1), torch.norm(y, dim=-1), mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return super(SampleSpeedMAE, self)._compute_std(torch.norm(y_hat, dim=-1), torch.norm(y, dim=-1))


class SpeedEnergyScore(EnergyScore): 
    """ EnergyScore on wind speed. """

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(EnergyScore, self)._check_mask(mask, val)

    def _compute_masked(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]):
        return super(SpeedEnergyScore, self)._compute_masked(torch.norm(y_hat, dim=-1, keepdim=True), 
                                                             torch.norm(y,     dim=-1, keepdim=True), 
                                                             mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return super(SpeedEnergyScore, self)._compute_std(torch.norm(y_hat, dim=-1), torch.norm(y, dim=-1))


class DirectionMAE(torch_metrics.MaskedMAE): 
    """ MAE on wind direction expressed as an angle in degrees. """

    def __init__(self, mask_nans=False, mask_inf=False, at=None, 
                 zerowind: float = 1e-2, **kwargs):
        super(DirectionMAE, self).__init__(mask_nans, mask_inf, at, **kwargs)
        self.zerowind = zerowind

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(DirectionMAE, self)._check_mask(mask, val)

    def _compute_masked(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]):
        phi = _compute_angle(y_hat, y, torch_functional.cosine_similarity)
        wind_mask = _mask_absent_wind(y, mask=mask, min_norm=self.zerowind) 
        return super(DirectionMAE, self)._compute_masked(phi, 0.0 * phi, wind_mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self._compute_masked(y_hat, y, mask=None)


class SampleDirectionMAE(SampleMetric):
    """ MAE on wind direction based on predicted samples. """
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(self, zerowind: float = 1e-2, **kwargs):
        super(SampleDirectionMAE, self).__init__(metric_fn=sampling_dir_mae, metric_fn_kwargs=kwargs.pop("metric_fn_kwargs", dict()), **kwargs)
        self.zerowind = zerowind

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(SampleDirectionMAE, self)._check_mask(mask, val)
    
    def _compute_masked(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]):
        wind_mask = _mask_absent_wind(y, mask=mask, min_norm=self.zerowind)
        return super(SampleDirectionMAE, self)._compute_masked(y_hat, y, wind_mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self._compute_masked(y_hat, y, mask=None)


class DirectionEnergyScore(EnergyScore): 
    """ Energy score on wind direction. """

    def __init__(self, *args, zerowind: float = 1e-2, **kwargs):
        super(DirectionEnergyScore, self).__init__(*args, **kwargs)
        self.zerowind = zerowind

    def _check_mask(self, mask: Optional[torch.Tensor], val: torch.Tensor):
        mask = _collapse_mask(mask, val)
        return super(EnergyScore, self)._check_mask(mask, val)

    def _compute_masked(self, y_hat: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]):
        wind_mask = _mask_absent_wind(y, mask=mask, min_norm=self.zerowind)
        phis = _compute_angle(y_hat, y.unsqueeze(self.sample_dim), torch_functional.cosine_similarity)
        phis = phis.unsqueeze(-1)  # as keepdim = True
        sl = [slice(None)] * phis.ndim # used to reduce on the sample_dim
        sl[self.sample_dim] = 0
        return super(DirectionEnergyScore, self)._compute_masked(phis, phis[sl] * 0.0, wind_mask)

    def _compute_std(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self._compute_masked(y_hat, y, mask=None)
