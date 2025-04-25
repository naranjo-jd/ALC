import numpy as np
import clean

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

def train(X, y, lr=0.01, epochs=1000, verbose=False):
    weights = np.zeros(X.shape[1])
    for i in range(epochs):
        grad = compute_gradient(X, y, weights)
        weights -= lr * grad
        if verbose and i % 100 == 0:
            loss = compute_loss(X, y, weights)
            print(f"Epoch {i}, Loss: {loss:.4f}")
    return weights

def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def cross_validate(X, y, folds, lr=0.01, epochs=1000):
    scores = []
    for train_idx, val_idx in folds:
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        weights = train(X_train, y_train, lr, epochs)
        y_pred = predict(X_val, weights)
        score = f1_score(y_val, y_pred)
        scores.append(score)
    return scores

def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def evaluar_mejores_transformaciones(X_raw, y, transformaciones):
    mejores_transformaciones = {}
    X_best = X_raw.copy()

    for i in range(X_raw.shape[1]):
        mejor_f1 = -1
        mejor_nombre = "identity"

        for nombre, f in transformaciones.items():
            X_mod = X_raw.copy()
            X_mod[:, i] = f(X_mod[:, i])
            X_std = (X_mod - X_mod.mean(0)) / X_mod.std(0)
            X_biased = clean.add_bias(X_std)

            f1s = []
            for train_idx, val_idx in clean.k_fold_split(X_biased, y, k=5):
                X_train, y_train = X_biased[train_idx], y[train_idx]
                X_val, y_val = X_biased[val_idx], y[val_idx]
                w = train(X_train, y_train)
                preds = predict(X_val, w)
                f1s.append(f1_score(y_val, preds.round()))

            avg_f1 = np.mean(f1s)
            if avg_f1 > mejor_f1:
                mejor_f1 = avg_f1
                mejor_nombre = nombre

        mejores_transformaciones[i] = mejor_nombre
        X_best[:, i] = transformaciones[mejor_nombre](X_raw[:, i])

    # Normalizar y añadir bias para salida final
    X_best = (X_best - X_best.mean(0)) / X_best.std(0)
    X_final = clean.add_bias(X_best)

    return X_final, mejores_transformaciones