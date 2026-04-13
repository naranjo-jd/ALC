import time
import numpy as np
import pandas as pd
import seaborn as sns
import kmeans
import metric
import matplotlib.pyplot as plt

# Definir métricas
def get_metrics(A):
    metrics = {
        "Euclidean": metric.euclidean,
        "Manhattan": metric.manhattan,
        "Chebyshev": metric.chebyshev,
        "Mahalanobis": metric.create_mahalanobis(A)
    }
    return metrics

def analysis(data, k, A, seed=42):
    if A.shape[0] != data.shape[1]:
        raise ValueError(f"La matriz A tiene tamaño {A.shape}, pero los datos son {data.shape[1]}D")
    results = []
    metrics = get_metrics(A)
    for name, metric_func in metrics.items():

        # Ejecutar K-Means con la métrica actual
        centroids, clusters = kmeans.Kmeans(data, k, metric_func, seed)

        # Calcular inercia con la métrica usada
        inertia = metric.inertia(clusters, centroids, metric_func)

        # ⚠️ Aquí incluimos la seed en los resultados (antes no estaba)
        results.append((seed, name, inertia))

    return results

def analysis_multiple_seeds(data, k, A, seeds):
    """Ejecuta el análisis para varias seeds."""
    all_results = []
    for seed in seeds:
        all_results.extend(analysis(data, k, A, seed))
    return all_results

def compare_inertia_by_seed(results):
    """Grafica la inercia de cada métrica según la seed en un barplot agrupado."""
    df_results = pd.DataFrame(results, columns=["Seed", "Métrica", "Inercia"])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="Seed", y="Inercia", hue="Métrica")

    plt.xlabel("Seed")
    plt.ylabel("Inercia")
    plt.title("Comparación de inercia por métrica y seed")
    plt.legend(title="Métrica")
    plt.grid(True)
    plt.show()