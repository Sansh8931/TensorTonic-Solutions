import numpy as np

sum=0

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    sum=0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                sum+=A[i][j]
            else:
                continue 

    
    return sum
