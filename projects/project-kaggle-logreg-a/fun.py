import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, weights):
    return sigmoid(np.dot(X, weights))

def predict(X, weights, threshold=0.5):
    probs = predict_proba(X, weights)
    return (probs >= threshold).astype(int)

def compute_loss(X, y, weights):
    m = len(y)
    h = predict_proba(X, weights)
    h = np.clip(h, 1e-10, 1 - 1e-10)  # Para evitar log(0)
    return -(1/m) * (np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h)))

def compute_gradient(X, y, weights):
    m = len(y)
    h = predict_proba(X, weights)
    return (1/m) * np.dot(X.T, (h - y))