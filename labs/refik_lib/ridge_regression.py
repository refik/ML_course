import numpy as np

def ridge_regression(y, tx, lambda_):
    a  = tx.T @ tx
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    b  = tx.T @ y
    return np.linalg.solve(a + aI, b)