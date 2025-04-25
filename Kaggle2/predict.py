import pandas as pd
import numpy as np
import clean
import model

# 1. Cargar datos de entrenamiento para calcular media y std
train_df = pd.read_csv("data/train_df.csv")
X_train, y_train, media, std = clean.prep(train_df)

# 2. Entrenar modelo con todos los datos
weights = model.train(X_train, y_train, lr=0.1, epochs=1000)

# 3. Cargar datos de test
test_df = pd.read_csv("data/train_df.csv")
paciente_ids = test_df["paciente_id"].values
y_true = test_df["target"].values

# 4. Preprocesar test (sin target)
def prep_test(df, media, std):
    df = df.copy()
    df = df.drop(columns=["paciente_id"])
    df["genero"] = df["genero"].map({"F": 0, "M": 1})
    X = df.drop(columns=["target"]).values
    X_normalizado = (X - media) / std
    X_final = clean.add_bias(X_normalizado)
    return X_final

X_test = prep_test(test_df, media, std)

# 5. Predecir
y_pred = model.predict(X_test, weights)

# 6. Crear y guardar DataFrame
predicciones = pd.DataFrame({
    "paciente_id": paciente_ids,
    "target": y_pred.astype(int)
})
predicciones.to_csv("predicciones.csv", index=False)
print("Archivo guardado como predicciones.csv")

# Calcular F1
f1 = model.f1_score(y_true, y_pred)
print("F1-Score en 'test':", f1)