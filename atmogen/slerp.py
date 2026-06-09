import torch
import numpy as np

def slerp(val, low, high):
    """
    Spherical linear interpolation.
    val: float between 0 and 1
    low: tensor of shape (..., d)
    high: tensor of shape (..., d)
    """
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    
    # Compute dot product
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True)
    
    # Clamp dot product to avoid numerical issues
    dot = torch.clamp(dot, -0.99999, 0.99999)
    
    omega = torch.acos(dot)
    so = torch.sin(omega)
    
    # If the angle is very small, fallback to linear interpolation
    if torch.any(so == 0):
        return (1.0 - val) * low + val * high
    
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high
