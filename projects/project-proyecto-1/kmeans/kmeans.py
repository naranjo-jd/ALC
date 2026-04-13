import numpy as np

### i) Inicialización de centroides
def centroids(data, k, seed=42):
    np.random.seed(seed)
    cols = data.columns  # Obtiene todas las columnas del DataFrame
    centroids = np.array([
        [np.random.uniform(data[col].min(), data[col].max()) for col in cols]
        for _ in range(k)
    ])
    return centroids

### ii) Asignación de clusters
def cluster(data, centroids, metric):
    clusters = [[] for _ in range(len(centroids))]
    points = data.to_numpy()  # Convierte el DataFrame en una matriz NumPy

    for point in points:
        # Calcula la distancia de cada punto a cada centroide
        distances = [metric(centroid, point)[0] for centroid in centroids]
        # Encuentra el índice del centroide más cercano
        nearest_index = np.argmin(distances)
        clusters[nearest_index].append(point)

    return clusters

### iii) Actualización de centroides
def update_centroids(clusters):
    if not clusters:
        return np.array([])  # Retorna un array vacío si no hay clusters

    dim = len(clusters[0][0]) if clusters[0] else 0  # Determina la dimensión automáticamente

    new_centroids = np.zeros((len(clusters), dim))

    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            new_centroids[i] = np.mean(cluster, axis=0)
        else:
            new_centroids[i] = np.zeros(dim)  # En caso de clusters vacíos

    return new_centroids

### iv) Algoritmo K-Means
def Kmeans(data, k, metric, seed = 42, max_iter=100, tol=1e-4):
    cent = centroids(data, k, seed)  # Inicializar centroides
    for _ in range(max_iter):
        clusters = cluster(data, cent, metric)
        new_centroids = update_centroids(clusters)

        # Comparación de cambios en centroides
        # distances = [metric(new_centroids[i], cent[i])[0] for i in range(len(cent))]
        distances = [np.linalg.norm(new_centroids[i] - cent[i]) for i in range(len(cent))]
        if max(distances) < tol:
            break
        cent = new_centroids

    return cent, clusters