import numpy as np


def tanh(X, W, b):
    sample_size, num_samples = X.shape
    return np.tanh(W * X + np.repeat(b, num_samples).reshape(sample_size, num_samples))


def grad_tanh(X, W, b):
    a = tanh(X, W, b)
    return np.ones(a.shape) - a ** 2
