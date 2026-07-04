import random
import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x=np.array(x)
    if not 0.0 <=p<=1.0:
        raise ValueError("Dropout probability  p must be in [0.0,1.0).")


    # Random Generator
    rand=rng.random(x.shape) if rng is not None else np.random.random(x.shape)


    #Keep mask:True if rand >=p (i.e.,kept neuron)
    keep_mask=rand>=p

    scale=1.0/(1.0-p) if p < 1.0 else 0.0
    dropout_pattern=keep_mask.astype(x.dtype)*scale

    #Apply Dropout
    output=x*dropout_pattern

    return output,dropout_pattern
    

    
        
       