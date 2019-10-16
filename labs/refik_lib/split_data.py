import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    
    returns x and y with tr for training, te for testing
    
    x_tr, x_te, y_tr, y_te
    """
    # set seed
    np.random.seed(seed)
    assert(x.shape[0] == y.shape[0])
    rindex = np.random.permutation(range(x.shape[0]))
    cutoff = int(len(rindex) * ratio)
    return (x.take(rindex)[0:cutoff], x.take(rindex)[cutoff:], 
            y.take(rindex)[0:cutoff], y.take(rindex)[cutoff:])