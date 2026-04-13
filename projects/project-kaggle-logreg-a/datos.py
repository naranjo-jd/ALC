import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Cargar datos
df_train = pd.read_csv("data/train_df.csv")
df_test = pd.read_csv("data/test_df.csv")

# 2. Eliminar columnas no numéricas que no sirven como features
df_train_ids = df_train["paciente_id"]
df_train = df_train.drop(columns=["paciente_id"])
df_test_ids = df_test["paciente_id"]  # guardar los IDs para el archivo final
df_test = df_test.drop(columns=["paciente_id"])

# 3. One-hot encoding para variables categóricas (como 'genero')
df_train = pd.get_dummies(df_train, drop_first=True)
df_test = pd.get_dummies(df_test, drop_first=True)

# 4. Alinear columnas entre train y test
df_test = df_test.reindex(columns=df_train.columns.drop("target"), fill_value=0)

# 5. Separar variables predictoras y target, y convertir a float64
X_train = df_train.drop(columns=["target"]).values.astype(np.float64)
y_train = df_train["target"].values.reshape(-1, 1).astype(np.float64)
X_test = df_test.values.astype(np.float64)

# 6. Escalar variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

__all__ = ["X_train", "y_train", "X_test", "df_test_ids"]