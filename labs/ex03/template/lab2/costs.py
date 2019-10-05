# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""

def compute_loss(y, tx, w, mae=False):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx @ w
    
    if mae:
        return np.mean(np.abs(e))
    else:
        return (np.transpose(e) @ e) / len(y)