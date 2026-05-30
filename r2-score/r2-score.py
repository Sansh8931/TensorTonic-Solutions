import numpy as np

def r2_score(y_true, y_pred):
    """
    Compute R² (coefficient of determination) for 1D regression.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        R² score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Edge case: constant target
    if np.all(y_true == y_true[0]):
        return float(np.allclose(y_true, y_pred))

    # Total sum of squares (variance relative to mean)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)

    # Residual sum of squares (error of predictions)
    ss_res = np.sum((y_true - y_pred)**2)

    return float(1 - ss_res / ss_tot)
