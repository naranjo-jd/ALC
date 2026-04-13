import numpy as np
from fun import compute_loss, compute_gradient, predict

def train(X, y, lr=0.01, epochs=1000, verbose=False):
    n_features = X.shape[1]
    weights = np.zeros(n_features)

    for i in range(epochs):
        grad = compute_gradient(X, y, weights)
        weights -= lr * grad

        if verbose and i % 100 == 0:
            loss = compute_loss(X, y, weights)
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return weights