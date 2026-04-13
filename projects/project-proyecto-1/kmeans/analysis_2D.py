import pandas as pd
import numpy as np
import kmeans
import graphers as grph
import metric
import analytics

data_2d = pd.read_csv("data/data_2d.csv")
data_2d_to_fit = data_2d.iloc[:, :-1]
k = 5
seed = 41
seeds = [41,40,21,44,50]
A = np.array([[2, 1], 
              [1, 2]])
metrica = metric.euclidean

init_centroids = kmeans.centroids(data_2d_to_fit, k, seed)
centroids, clusters = kmeans.Kmeans(data_2d_to_fit, k, metrica, seed)  # Ahora obtenemos clusters
grph.plot_2d(data_2d_to_fit, init_centroids)
grph.plot_2d(data_2d_to_fit, centroids)

analisis = analytics.analysis_multiple_seeds(data_2d_to_fit, k, A, seeds)
analytics.compare_inertia_by_seed(analisis)