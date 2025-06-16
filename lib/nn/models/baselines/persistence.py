from typing import Optional

import torch
from torch import Tensor, nn

import tsl.nn.models as tsl_models


class PersistenceModel(tsl_models.BaseModel):
    """
    A persistence model for time series forecasting that repeats values from a past temporal point 
    to predict future values.
    If `lag == 1`, it simply repeats the last observed time step across the entire forecasting horizon.  
    If `lag > 1`, it predicts the value at time h in the forecasting horizon 
    looking at the values at time h - lag to capture simple seasonal patterns.

    To generate Monte Carlo samples for probabilistic forecasting, white noise is added to the output. 
    The noise has a standard deviation equal to the temporal variability (computed as the std dev over time).

    Note:
    - If `lag` is greater than the input window length, the model raises an error.
    - If the lag is shorter than the forecast horizon, it will be tiled to fit (not encouraged).
    """
    def __init__(self, horizon: int, lag: int = 1):
        """
        lag > 1: length of the seasonal period - delay between predictive window and predicted horizon
        lag = 1 consider the last observations and replicates it as prediction for the entire horizon 
        """
        super().__init__()
        self.horizon = horizon
        self.lag = lag if lag > 0 else - lag
        
    def forward(self, x: Tensor, mc_samples: Optional[int] = None) -> Tensor:
        # shape x: b t n f
        # shape y: b h n f
        t = x.size(1)
        mu = x[:, t - self.lag: t - self.lag + self.horizon]
        if self.lag > t:
            raise NotImplementedError("lag exceed the input window")
        if mu.size(1) < self.horizon:  # mu does not fill the horizon
            # repeats to fill the horizon
            if mu.size(1) > 1: # the code works even for mu.size(1) > 1, but perhaps not as inteded in this case
                import warnings
                warnings.warn("If there is a gap between the predictive window and horizon, the PersistenceModel may not behave as expected.", RuntimeWarning)
            mu = torch.cat([mu]*(self.horizon // mu.size(1) + 1), axis=1)[:, :self.horizon]
        if mc_samples is None:
            return mu
        sigma = x.std(dim=1, keepdim=True)
        noise_shape = (mc_samples, x.size(0), self.horizon, *x.shape[2:])
        return mu + sigma * torch.randn(*noise_shape, device=x.device)