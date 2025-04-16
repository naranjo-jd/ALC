import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(X, y, weights):
    m = X.shape[0]
    z = X @ weights
    h = sigmoid(z)
    epsilon = 1e-15
    loss = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return loss

def train(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros((n, 1))
    losses = []

    for epoch in range(epochs):
        z = X @ weights
        h = sigmoid(z)
        gradient = (1/m) * (X.T @ (h - y))
        weights -= lr * gradient

        if epoch % 100 == 0:
            loss = compute_loss(X, y, weights)
            losses.append(loss)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return weights

def predict(X, weights, threshold=0.5):
    probs = sigmoid(X @ weights)
    return (probs >= threshold).astype(int)