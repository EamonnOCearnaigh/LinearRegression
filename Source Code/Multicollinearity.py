import numpy as np


def make_regression_multivariate(
    n_samples,
    n_uncorrelated,
    n_correlated,
    correlation,
    weights,
    bias,
    noise=1,
    seed=42,
):
    np.random.seed(seed)

    X_correlated = None
    if n_correlated > 0:
        cov = correlation * np.ones((n_correlated, n_correlated)) + (
            1 - correlation
        ) * np.eye(n_correlated)

        X_correlated = np.random.multivariate_normal(
            mean=np.zeros(n_correlated),
            cov=cov,
            size=n_samples,
        )

    X_uncorrelated = None
    if n_uncorrelated > 0:
        X_uncorrelated = np.random.multivariate_normal(
            mean=np.zeros(n_uncorrelated), cov=np.eye(n_uncorrelated), size=n_samples
        )

    X = None
    if n_uncorrelated <= 0:
        X = X_correlated
    elif n_correlated <= 0:
        X = X_uncorrelated
    else:
        X = np.hstack([X_correlated, X_uncorrelated])

    e = np.random.normal(loc=0, scale=noise, size=n_samples)
    y = bias + np.dot(X, weights) + e

    return X, y
