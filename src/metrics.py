import numpy as np

def compute_mse(y_true, y_pred):
    # Mean Squared Error: average of squared differences
    # Measures how well predictions match the true labels
    return np.mean((y_true - y_pred) ** 2)


def compute_norm(w):
    # L2 norm of weight vector: sqrt(sum of squares)
    # Measures the magnitude of the learned weights
    return np.linalg.norm(w)
