import numpy as np
from . import models, metrics

# Train OLS across different dimensions and measure errors
def run_ols_experiment(X_train_full, y_train, X_test_full, y_test, d_values):
    train_errors = []
    test_errors = []

    for d in d_values:
        X_tr = X_train_full[:, :d]
        X_te = X_test_full[:, :d]

        w_hat = models.fit_ols(X_tr, y_train)
        y_pred_tr = X_tr @ w_hat
        y_pred_te = X_te @ w_hat

        train_errors.append(metrics.compute_mse(y_train, y_pred_tr))
        test_errors.append(metrics.compute_mse(y_test, y_pred_te))

    return train_errors, test_errors

# Train Ridge regression across different dimensions
def run_ridge_experiment(X_train_full, y_train, X_test_full, y_test, d_values, alpha=0.1):
    train_errors = []
    test_errors = []

    for d in d_values:
        X_tr = X_train_full[:, :d]
        X_te = X_test_full[:, :d]

        w_hat = models.fit_ridge(X_tr, y_train, alpha=alpha)
        y_pred_tr = X_tr @ w_hat
        y_pred_te = X_te @ w_hat

        train_errors.append(metrics.compute_mse(y_train, y_pred_tr))
        test_errors.append(metrics.compute_mse(y_test, y_pred_te))

    return train_errors, test_errors

# Train minimum norm interpolator and track weight magnitude
def run_min_norm_experiment(X_train_full, y_train, X_test_full, y_test, d_values):
    train_errors = []
    test_errors = []
    norms = []

    for d in d_values:
        X_tr = X_train_full[:, :d]
        X_te = X_test_full[:, :d]

        w_hat = models.fit_min_norm(X_tr, y_train)
        y_pred_tr = X_tr @ w_hat
        y_pred_te = X_te @ w_hat

        train_errors.append(metrics.compute_mse(y_train, y_pred_tr))
        test_errors.append(metrics.compute_mse(y_test, y_pred_te))
        norms.append(metrics.compute_norm(w_hat))

    return train_errors, test_errors, norms

# Train with gradient descent and compare to closed-form solutions
def run_gradient_descent_experiment(X_train_full, y_train, X_test_full, y_test, d_values, learning_rate=0.01, epochs=5000):
    train_errors = []
    test_errors = []

    for d in d_values:
        X_tr = X_train_full[:, :d]
        X_te = X_test_full[:, :d]

        w_hat = models.fit_gradient_descent(X_tr, y_train, learning_rate=learning_rate, epochs=epochs)
        y_pred_tr = X_tr @ w_hat
        y_pred_te = X_te @ w_hat

        train_errors.append(metrics.compute_mse(y_train, y_pred_tr))
        test_errors.append(metrics.compute_mse(y_test, y_pred_te))

    return train_errors, test_errors
