import numpy as np

def least_squares(y, tx):
    """Calculating weights for the closed form solution of 
    minimizing mean squared errors.
    This is essentially linear regression.
    Returns weights as a column  vector.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)