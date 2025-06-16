from typing import Literal, Optional

import torch
from torch import Tensor, nn
from einops import rearrange, einsum

from tsl.nn.layers.multi.linear import MultiLinear
from tsl.nn.utils import get_layer_activation


class AdditiveGaussianReadoutLayer(torch.nn.Module):
    """ 
    Samples are drawn from a Gaussian distribution N(mu, Sigma), where:
     - mu[i...j] = x[i...j] for all indices i...j, and 
     - Sigma[i...j] = PP^T with P a learnable matrix of shape 
        [num_feats, num_feats] shared across all indices i...j.
    """

    def __init__(self, *args, input_size, output_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, mc_samples: Optional[int] = None) -> Tensor:
        if mc_samples is None:
            return x       
        eps = torch.randn(mc_samples, *x.shape, device=x.device)
        return x[None, ...] + self.lin(eps) 


class GaussianReadoutLayer(torch.nn.Module):
    """ 
    Samples are drawn from a Gaussian distribution N(mu, Sigma), where:
     - mu[i...j] = Linear1(x[i...j]) and 
     - Sigma[i...j] = P[i...j]P[i...j]^T, with P[i...j] = Linear2(x[i...j]).
    """
    
    def __init__(self, *args, input_size, output_size, **kwargs):
        super().__init__(*args, **kwargs)
        # y_samp = mu(x) + sigma(x) * eps
        # [m, ..., f] = [..., f] + [..., f, f] * [m, ..., f]
        self.output_size = output_size
        self.mu = nn.Linear(in_features=input_size, out_features=output_size)
        self.sigma = nn.Linear(in_features=input_size, out_features=output_size**2)

    def forward(self, x, mc_samples: Optional[int] = None) -> Tensor:
        mu = self.mu(x) 
        if mc_samples is None:
            return mu
        sigma = self.sigma(x)
        sigma = rearrange(sigma, "... (l f) -> ... l f", f=mu.size(-1))
        eps = torch.randn(mc_samples, *mu.shape, device=mu.device)
        out = mu.unsqueeze(0) 
        out = out + einsum(sigma, eps, "... l f, m ... f -> m ... l")
        return out


class MultiGaussianReadoutLayer(torch.nn.Module):
    """ 
    Samples are drawn from a Gaussian distrubution N(mu, Sigma), where:
     - mu[i...j] = Linear1[i...j](x[i...j]), and 
     - Sigma[i...j] = P[i...j]P[i...j]^T, with 
        P[i...j] = Linear2[i...j](x[i...j]).
    In this formulation, Linear1[i...j] and Linear2[i...j] denote independently
    parameterized linear layers at each index i...j.
    """

    def __init__(self, *args, input_size, output_size, n_nodes, horizon, **kwargs):
        super().__init__(*args, **kwargs)
        # y_samp = mu(x) + sigma(x) * eps
        # [m, ..., f] = [..., f] + [..., f, f] * [m, ..., f]
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.output_size = output_size
        self.mu = MultiLinear(in_channels=input_size, 
                              out_channels=output_size, 
                              n_instances=n_nodes * horizon,
                              pattern='b o f', instance_dim='o')  # batch, (horizon, n_nodes), feats
        self.sigma = MultiLinear(in_channels=input_size, 
                                 out_channels=output_size**2, 
                                 n_instances=n_nodes * horizon,
                                 pattern='b o f', instance_dim='o')  # batch, (horizon, n_nodes), feats

    def forward(self, x, mc_samples: Optional[int] = None) -> Tensor:
        x_ = rearrange(x, "... h n f -> ... (h n) f")
        mu_ = self.mu(x_) 
        # mu = rearrange(mu, "... n (h f) -> ... h n f", h=self.horizon)       
        mu = rearrange(mu_, "... (h n) f -> ... h n f", h=self.horizon)
        if mc_samples is None:
            return mu
        sigma_ = self.sigma(x_)
        # sigma = rearrange(sigma, "... n (h l f) -> ... h n l f", f=self.output_size, h=self.horizon)
        sigma = rearrange(sigma_, "... (h n) (l f) -> ... h n l f", f=self.output_size, h=self.horizon)
        eps = torch.randn(mc_samples, *mu.shape, device=mu.device)
        out = mu.unsqueeze(0) 
        out = out + einsum(sigma, eps, "... l f, m ... f -> m ... l")
        return out
    


class SamplingReadoutLayer(torch.nn.Module):
    """ 
    Readout layer that outputs a Gaussian distribution.

    This layer takes the activations `x` of shape [..., num_feats] from the 
    preceding layer and produces Monte Carlo samples with shape 
    [mc_sample, ..., num_feats].

    The `noise_mode` argument controls the specific parameterization used for 
    the Gaussian distribution.
    """

    def __init__(self, *args, 
                 input_size: int, 
                 output_size: int,
                 n_nodes: int, 
                 horizon: int, 
                 noise_mode: Literal["lin", "multi", "add", "none"] = None,
                 pre_activation: str = 'elu',
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = get_layer_activation(pre_activation)()
    
        if noise_mode == "none":
            noise_mode = None
        self.noise_mode = noise_mode

        if noise_mode == "none":
            self.sample_decoder = None
        elif noise_mode == "add":
            self.sample_decoder = AdditiveGaussianReadoutLayer(input_size=input_size, output_size=output_size)
        elif noise_mode == "lin":
            self.sample_decoder = GaussianReadoutLayer(input_size=input_size, output_size=output_size)
        elif noise_mode == "multi":
            self.sample_decoder = MultiGaussianReadoutLayer(input_size=input_size, output_size=output_size, n_nodes=n_nodes, horizon=horizon)
        else:
            raise NotImplementedError

    def forward(self, x, mc_samples: Optional[int] = None) -> Tensor:
        x = self.activation(x)
        out = self.sample_decoder(x, mc_samples)
        return out
