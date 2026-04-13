import pandas as pd
import clustering as clust
import graphers as grph
import metric

data_2d = pd.read_csv("data/data_2d.csv")
data_3d = pd.read_csv("data/data_3d.csv")

# Test 2D
centros = clust.centroids(data_2d, 2, 5)
centroides, clusters_2d = clust.kmeans(data_2d, 2, 5, metric.euclidean)  # Ahora obtenemos clusters
grph.plot_2d(data_2d, centros)
grph.plot_2d(data_2d, centroides)

print("Inercia 2d:", metric.inertia(clusters_2d, centroides, metric.euclidean))  # Usamos los clusters correctos

# Test 3D
centros = clust.centroids(data_3d, 3, 5)
centroides, clusters_3d = clust.kmeans(data_3d, 3, 5, metric.euclidean)  # Ahora obtenemos clusters
grph.plot_3d(data_3d, centros)
grph.plot_3d(data_3d, centroides)

print("Inercia 3d:", metric.inertia(clusters_3d, centroides, metric.euclidean))  # Usamos los clusters correctos