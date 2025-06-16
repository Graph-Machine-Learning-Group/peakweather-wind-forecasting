import torch
from torch import Tensor, median
from torch.nn import functional as torch_functional

from lib.metrics.sample_metrics import _check_input_shape


def sampling_MAE(preds: Tensor, target: Tensor, reduction: str = "none") -> Tensor:
    """Computes the mean absolute error (MAE) for a set of predictive samples of shape 
    [samples, *target_shape] and a single target. 

    Args:
        preds (Tensor): The predictive samples.
        target (Tensor): The target values.
        reduction (str, optional): A pytorch-compatible reduction. Defaults to "none".

    Returns:
        Tensor: The computed MAE with desired reduction.
    """    
    preds, target = _check_input_shape(preds, target)
    return torch_functional.l1_loss(input=median(preds, dim=0)[0], target=target, reduction=reduction)

def sampling_MSE(preds: Tensor, target: Tensor, reduction: str = "none") -> Tensor:
    """Computes the mean squared error (MSE) for a set of predictive samples of shape 
    [samples, *target_shape] and a single target. 

    Args:
        preds (Tensor): The predictive samples.
        target (Tensor): The target values.
        reduction (str, optional): A pytorch-compatible reduction. Defaults to "none".

    Returns:
        Tensor: The computed MSE with desired reduction.
    """
    preds, target = _check_input_shape(preds, target)
    return torch_functional.mse_loss(input=preds.mean(axis=0), target=target, reduction=reduction)

def sampling_cosine_similarity(preds: Tensor, target: Tensor, dim: int = -1) -> Tensor:
    """Computes the cosine similarity for a set of predictive samples of shape 
    [samples, *target_shape] and a single target. 

    Args:
        preds (Tensor): The predictive samples.
        target (Tensor): The target values.
        dim (int, optional): The dimension along which the similarity is computed. Defaults to -1.

    Returns:
        Tensor: The computed cosine similarity.
    """
    preds, target = _check_input_shape(preds, target)
    preds_norm = torch.nn.functional.normalize(preds, p=2.0, dim=dim) 
    y_hat = preds_norm.mean(0)  
    # similarity of the extrinsic mean with the target
    return torch_functional.cosine_similarity(x1=y_hat, x2=target, dim=dim) 
