import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    
    """

    # Ensure inputs are Numpy arrays for vectorized Operations
    param=np.array(param,dtype=float)
    m=np.array(m,dtype=float)
    v=np.array(v,dtype=float)
    grad=np.array(grad,dtype=float)
    
    # Step 1:Update first Moment
    
    m_new=beta1*m+(1-beta1)*grad

    
   # Update Second Moment
    v_new=v*beta2+(1-beta2)*(grad**2)

    # Step 3 : Bias Correction
    m_hat=m_new/(1-beta1 ** t )
    v_hat=v_new/(1-beta2** t)

    # Step 4: Parameter Update
    param_new=param-lr*m_hat/(np.sqrt(v_hat)+eps)


    return param_new,m_new,v_new

     



    