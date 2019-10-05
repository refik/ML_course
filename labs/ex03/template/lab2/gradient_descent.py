# -*- coding: utf-8 -*-
from costs import compute_loss
import numpy as np

"""Gradient Descent"""

def compute_gradient(y, tx, w, mae=False):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and error vector
    # ***************************************************
    e = y - tx @ w
    
    if mae:
        return -1/len(y) * tx.T @ np.sign(e)
    else:
        return -1/len(y) * tx.T @ e


def gradient_descent(y, tx, initial_w, max_iters, gamma, mae=False):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w, mae=mae)]
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        gradient = compute_gradient(y, tx, w, mae)

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w, mae=mae)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws