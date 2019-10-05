# -*- coding: utf-8 -*-
from costs import compute_loss
from gradient_descent import compute_gradient
from helpers import batch_iter
"""Stochastic Gradient Descent"""

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w
    for y_batch, tx_batch in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(y, tx, w)

        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent: loss={l}, w0={w0}, w1={w1}".format(
              l=loss, w0=w[0], w1=w[1]))

    return losses, ws