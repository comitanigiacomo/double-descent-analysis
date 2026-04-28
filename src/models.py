import numpy as np

# OLS: minimize sum of squared residuals
def fit_ols(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Ridge: add penalty on weight magnitude
def fit_ridge(X, y, alpha=0.1):
    d = X.shape[1]
    return np.linalg.inv(X.T @ X + alpha * np.eye(d)) @ X.T @ y

# Minimum norm interpolator from Belkin et al. framework
# For d <= n: use standard OLS
# For d > n: w = X^T (X X^T)^(-1) y
# Finds shortest weight vector that perfectly fits the training data
def fit_min_norm(X, y):
    n, d = X.shape

    if d <= n:
        return np.linalg.inv(X.T @ X) @ X.T @ y
    else:
        return X.T @ np.linalg.inv(X @ X.T) @ y


def fit_gradient_descent(X, y, learning_rate=0.01, epochs=5000):
    n, d = X.shape
    w = np.zeros(d)

    for _ in range(epochs):
        predictions = X @ w
        gradient = (2 / n) * X.T @ (predictions - y)
        w -= learning_rate * gradient

    return w
