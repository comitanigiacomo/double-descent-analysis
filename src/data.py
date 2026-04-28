import numpy as np


def generate_synthetic_dataset(n, d, noise_level=1.0, seed=30):
    generator = np.random.default_rng(seed)

    # True weights: only first 20 features are active
    # Extra features are noise that the model should not learn
    w = generator.standard_normal(d)
    if d > 20:
        w[20:] = 0

    # Generate train and test independently
    X_train = generator.standard_normal((n, d))
    X_test = generator.standard_normal((n, d))

    # Independent noise for each set
    noise_train = generator.standard_normal(n) * noise_level
    noise_test = generator.standard_normal(n) * noise_level

    # Labels from true linear model: y = X @ w + noise
    y_train = X_train @ w + noise_train
    y_test = X_test @ w + noise_test

    return X_train, y_train, X_test, y_test
