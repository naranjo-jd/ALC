import pandas as pd
import numpy as np
import kmeans
import graphers as grph
import metric
import analytics

data_3d = pd.read_csv("data/data_3d.csv")
data_3d_to_fit = data_3d.iloc[:, :-1]
k = 5
seed = 41
seeds = [41,40,21,44,50]
A = np.array([
    [3, 1, 1],
    [1, 3, 1],
    [1, 1, 3]
])
metrica = metric.euclidean

init_centroids = kmeans.centroids(data_3d_to_fit, k, seed)
centroids, clusters = kmeans.Kmeans(data_3d_to_fit, k, metrica, seed)  # Ahora obtenemos clusters
grph.plot_3d(data_3d_to_fit, init_centroids)
grph.plot_3d(data_3d_to_fit, centroids)

analisis = analytics.analysis_multiple_seeds(data_3d_to_fit, k, A, seeds)
analytics.compare_inertia_by_seed(analisis)