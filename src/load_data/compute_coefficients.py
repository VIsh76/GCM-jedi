import numpy as np
import torch

def _unflatten(coef, num_dim, dim):
    """Given a 1D array return an array with size of ones
    And the dim id corresponding

    Args:
        coef (array): 1d coef
        num_dim (int): length of size of the final vec
        dim (int): id of the dimension that matches the coef

    Returns:
        array: vector that can be added or multiply to the output
    """
    for _ in range(num_dim - 1):
        coef = np.expand_dims(coef, axis=-1)
    coef = np.swapaxes(coef, 0, dim)
    return coef

def compute_level_coef(levels, num_dim, dim):
    """
    Similar to Graphcast coefficients
    n_levels: number of levels
    num_dim: number of dim of the target/input
    dim: dimension id of the coefficient (all other are 1)
    """
    coef = np.arange(1, 1+len(levels)) / 0.65
    coef = _unflatten(coef, num_dim, dim)
    return coef

def compute_lat_coef(latitudes, num_dim, dim):
    """
    Similar to Graphcast coefficients
    n_lats: number of lattitudes
    num_dim: number of dim of the target/input
    dim: dimension id of the coefficient (all other are 1)
    """
    coef = np.cos(latitudes)
    coef = coef / np.sum(coef) * len(latitudes)
    coef = _unflatten(coef, num_dim, dim)
    return coef
