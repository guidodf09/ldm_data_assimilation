import os
import numpy as np
import h5py  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import re
from pathlib import Path


def normalize_rates(rates, method="standard"):
    """
    Normalize rates (e.g., time series) over batch and time: [N, T, F]
    """
    if method == "standard":
        mean = rates.mean(dim=(0, 1), keepdim=True)
        std = rates.std(dim=(0, 1), keepdim=True) + 1e-6
        normalized = (rates - mean) / std
        stats = {'mean': mean, 'std': std}

    elif method == "minmax":
        min_val = rates.amin(dim=(0, 1), keepdim=True)
        max_val = rates.amax(dim=(0, 1), keepdim=True)
        range_val = max_val - min_val + 1e-6
        normalized = (rates - min_val) / range_val
        stats = {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError("Unsupported normalization method. Use 'standard' or 'minmax'.")
    
    return normalized, stats

def denormalize_rates(normalized_rates, stats, method="standard"):
    """
    Denormalize the normalized rates: [N, T, F]
    """
    if method == "standard":
        return normalized_rates * stats['std'] + stats['mean']
    elif method == "minmax":
        range_val = stats['max'] - stats['min'] + 1e-6
        return normalized_rates * range_val + stats['min']
    else:
        raise ValueError("Unsupported denormalization method. Use 'standard' or 'minmax'.")


def normalize_static_images(images, method="standard"):
    """
    Normalize static 3D image tensors over batch and spatial dims: [N, C, H, W, D]
    """
    if method == "standard":
        mean = images.mean(dim=(0, 2, 3, 4), keepdim=True)
        std = images.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-6
        normalized = (images - mean) / std
        stats = {'mean': mean, 'std': std}
    
    elif method == "minmax":
        min_val = images.amin(dim=(0, 2, 3, 4), keepdim=True)
        max_val = images.amax(dim=(0, 2, 3, 4), keepdim=True)
        range_val = max_val - min_val + 1e-6
        normalized = (images - min_val) / range_val
        stats = {'min': min_val, 'max': max_val}
    
    else:
        raise ValueError("Unsupported normalization method. Use 'standard' or 'minmax'.")
    
    return normalized, stats

def denormalize_static_images(normalized_images, stats, method="standard"):
    """
    Denormalize the static image tensors: [N, C, H, W, D]
    """
    if method == "standard":
        return normalized_images * stats['std'] + stats['mean']
    elif method == "minmax":
        range_val = stats['max'] - stats['min'] + 1e-6
        return normalized_images * range_val + stats['min']
    else:
        raise ValueError("Unsupported denormalization method. Use 'standard' or 'minmax'.")

        
        
def model2tricat(model, thresh1, thresh2):
    
    '''
    Tricategorical-ize the geo-model, given a thresholds.

    Parameters
    ----------
    model: ndarray
        The multidimensional array of geo-models (can have any shape).
    thresh1: float
        Threshold used in facies0-facies1.
    thresh2: float
        Threshold used in facies1-facies2.

    Returns
    ------
    c_d: ndarray:
        Binarized geomodel
    '''
    
    model_copy = np.copy(model)
    model_copy[model < thresh1] = 0.
    model_copy[(model >= thresh1) & (model <= thresh2)] = 0.5
    model_copy[model > thresh2] = 1.
    
    return model_copy




    
