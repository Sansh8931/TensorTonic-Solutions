import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    y=np.asarray(x,dtype=float)
    
    return 1/(1+np.exp(-y));