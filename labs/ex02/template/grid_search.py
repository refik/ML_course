# -*- coding: utf-8 -*-
""" Grid Search"""

from costs import compute_loss
import numpy as np


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    losses = np.zeros((len(w0), len(w1)))
    
    min_loss = float("inf")
    min_loss_w = ()
    
    for i in range(len(w0)):
        for j in range(len(w1)):
            loss_w  = (w0[i], w1[j])
            loss = compute_loss(y, tx, loss_w)
            losses[i,j] = loss
            if loss < min_loss:
                min_loss = loss
                min_loss_w = loss_w
                
    print("Minimum Loss: {} for w={}".format(min_loss, min_loss_w))
    
    return losses