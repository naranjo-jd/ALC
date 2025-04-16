import pandas as pd
import datos
import log_model

# Entrenamiento
weights = log_model.train(datos.X_train, datos.y_train, lr=0.1, epochs=1000)

# Predicción en test usando la nueva función
y_pred = log_model.predict(datos.X_test, weights).flatten()

# Guardar predicciones
df_resultado = pd.DataFrame({
    "paciente_id": datos.df_test_ids,
    "prediccion": y_pred
})
df_resultado.to_csv("predicciones.csv", index=False)
print("✅ Predicciones guardadas en predicciones.csv")