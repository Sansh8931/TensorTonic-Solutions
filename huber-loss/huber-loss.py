import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute mean Huber Loss for regression.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Shape check
    if y_true.shape != y_pred.shape:
        return None

    # Error
    error = y_true - y_pred
    abs_error = np.abs(error)

    # Piecewise definition
    loss = np.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta)
    )

    return float(np.mean(loss))
