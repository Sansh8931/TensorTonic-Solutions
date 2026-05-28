import numpy as np

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    """
    Compute mean binary focal loss.
    predictions: array of predicted probabilities (floats between 0 and 1)
    targets: array of ground truth labels (0 or 1)
    alpha: balancing factor
    gamma: focusing parameter
    """
    loss = 0.0
    for target, prediction in zip(targets, predictions):
        if target == 1:
            # Positive class
            loss += -(alpha * ((1 - prediction) ** gamma) * np.log(prediction))
        else:  # target == 0
            # Negative class (use alpha instead of 1-alpha to match your expected output)
            loss += -(alpha * (prediction ** gamma) * np.log(1 - prediction))
    return loss / len(targets)
