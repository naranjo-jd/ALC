import numpy as np

def normalizar(X):
    media = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - media) / std, media, std

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def k_fold_split(X, y, k=5, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    
    fold_size = len(y) // k
    folds = []
    
    for i in range(k):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        folds.append((train_idx, val_idx))
    
    return folds

def prep(data):
    df = data.copy()

    df = df.drop(columns=["paciente_id"])
    df["genero"] = df["genero"].map({"F": 0, "M": 1})

    y = df["target"].values
    X = df.drop(columns=["target"]).values

    X_normalizado, media, std = normalizar(X)
    X_final = add_bias(X_normalizado)

    return X_final, y, media, std