# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


def compute_mse(y, tx, w):
    """Calculate mean squared error loss"""
    e = y - tx @ w
    return (np.transpose(e) @ e) / len(y)
    
def compute_mae(y, tx, w):
    """Calculate loss for mean absolute deviation.
    
    Loss for sub gradient descent
    """
    e = y - tx @ w
    return np.mean(np.abs(e))