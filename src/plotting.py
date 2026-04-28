import matplotlib.pyplot as plt


def plot_ols_experiment(d_values, train_errors, test_errors, n):
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, test_errors, label='Test Error', color='darkorange', marker='o')
    plt.plot(d_values, train_errors, label='Train Error', color='blue', marker='s')
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(-1, 100)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Classical OLS Implementation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ridge_experiment(d_values, train_errors, test_errors, n):
    plt.figure(figsize=(10, 6))
    plt.plot(d_values, test_errors, label='Ridge Test Error', color='darkorange', marker='^')
    plt.plot(d_values, train_errors, label='Ridge Train Error', color='blue', marker='x', linestyle='--')
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(-1, 40)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Ridge Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ols_vs_ridge(d_values, ols_train, ols_test, ridge_train, ridge_test, n):
    plt.figure(figsize=(10, 6))

    plt.plot(d_values, ols_test, label='OLS Test Error (Crash)', color='purple', marker='o')
    plt.plot(d_values, ols_train, label='OLS Train Error', color='green', marker='s')
    plt.plot(d_values, ridge_test, label='Ridge Test Error', color='darkorange', marker='^')
    plt.plot(d_values, ridge_train, label='Ridge Train Error', color='blue', marker='x', linestyle='--')
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(-1, 100)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Classical OLS vs Ridge Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_double_descent(d_values, train_errors, test_errors, n):
    plt.figure(figsize=(10, 6))
    plt.yscale('log')

    plt.plot(d_values, test_errors, label='Test Error', color='darkorange', marker='D')
    plt.plot(d_values, train_errors, label='Train Error', color='blue', marker='x', linestyle='--')
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(0.5, 500)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Mean Squared Error (MSE) - Log Scale')
    plt.title('The Double Descent Risk Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_weights_norm(d_values, norms, n):
    plt.figure(figsize=(10, 6))
    plt.yscale('log')

    plt.plot(d_values, norms, label='Minimum Norm Interpolator (||w||)', color='blue', marker='s')
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(2, 50)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('L2 Norm of Weights (||w||) - Log Scale')
    plt.title('Weights Norm Behavior (Inductive Bias)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_closed_form_vs_gd(d_values, closed_form_errors, gd_errors, n):
    plt.figure(figsize=(10, 6))
    plt.yscale('log')

    plt.plot(d_values, closed_form_errors, label='Closed-Form Test Error', color='darkorange',
             marker='D', linewidth=2)
    plt.plot(d_values, gd_errors, label='Gradient Descent Test Error', color='blue',
             marker='x', linestyle='--', linewidth=2)
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.ylim(0.5, 500)
    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Mean Squared Error (MSE) - Log Scale')
    plt.title('Closed-Form vs Gradient Descent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_noise_effect(d_values, noise_scenarios, n):
    plt.figure(figsize=(10, 6))
    plt.yscale('log')

    plt.plot(d_values, noise_scenarios[0.2], label='Low Noise ($\\sigma=0.2$)',
             color='blue', marker='o', markersize=4)
    plt.plot(d_values, noise_scenarios[1.5], label='High Noise ($\\sigma=1.5$)',
             color='darkorange', marker='s', markersize=4)
    plt.axvline(x=n, color='black', linestyle='--', label='Interpolation Threshold (d=n)')

    plt.xlabel('Model Complexity (Dimension d)')
    plt.ylabel('Test MSE (Log Scale)')
    plt.title('Effect of Noise on the Double Descent Peak')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
