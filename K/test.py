import clean
import model
import numpy as np
import pandas as pd

data = pd.read_csv("data/train_df.csv")

paciente_ids = data["paciente_id"].values

X, y = clean.prep(data)
folds = clean.k_fold_split(X, y, k=5)

scores = model.cross_validate(X, y, folds, lr=0.1, epochs=1000)
print("F1-Scores por fold:", scores)
print("F1-Score promedio:", np.mean(scores))
