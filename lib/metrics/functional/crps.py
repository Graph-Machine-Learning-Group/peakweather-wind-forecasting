import torch


def energy_score(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ Computes the Energy Score (CRPS generalized to multivariate distributions).
    
    Implements:
    
    ES(F,y) = E(||X-y||) - 0.5 E(||X-X'||)
             X~F              X,X'~F

    where the expected values are approximated with MC samples. 
    
    Args:
        samples (torch.Tensor): MC samples of shape [M, ..., D] from the predictive distribution.
        target (torch.Tensor): The corresponding target vectors of shape [..., D]

    Returns:
        torch.Tensor: the energy score computed along the channel dimension.
    """
    assert samples.ndim == target.ndim + 1

    sample_dim = 0    
    mc_samples = samples.shape[sample_dim]
    
    # First term: E(||X-y||), X~F
    diff_to_tgt = samples - target.unsqueeze(sample_dim)
    eucl_distance = torch.norm(diff_to_tgt, dim=-1)  # across channels
    first_term = eucl_distance.mean(dim=sample_dim)  # across samples
    
    # Second term: E(||X-X'||), X,X'~F
    x_i = samples.unsqueeze(sample_dim)   # shape = [1, M, ..., D]
    x_j = samples.unsqueeze(sample_dim+1) # shape = [M, 1, ..., D]
    
    pairwise_eucl_distance = torch.norm(x_i - x_j, dim=-1)  # across channels, shape = [M, M, ...]
    second_term = pairwise_eucl_distance.sum(dim=(sample_dim, sample_dim+1)) / (mc_samples * (mc_samples-1)) # across samples, shape = [...]
    
    energy_score = first_term - 0.5 * second_term
    
    return energy_score # shape = [...] = target.shape[:-1], i.e., target shape w/ channel dim reduced
    