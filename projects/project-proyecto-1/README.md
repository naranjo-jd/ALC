# Project: Proyecto 1 (K-Means)

Implementation of K-Means clustering from scratch with multiple distance metrics and inertia-based comparison across random seeds.

## Structure

- `kmeans/kmeans.py`: centroid initialization, cluster assignment, centroid updates, and K-Means loop.
- `kmeans/metric.py`: Euclidean, Manhattan, Chebyshev, and Mahalanobis metrics plus inertia calculation.
- `kmeans/analytics.py`: metric/seed comparison helpers.
- `kmeans/analysis_2D.py`, `analysis_3D.py`, `analysis_Mall.py`: runnable analyses over provided datasets.
- `kmeans/data/`: datasets (`data_2d.csv`, `data_3d.csv`, `Mall_Customers.csv`).

## Run

```bash
cd kmeans
python3 analysis_2D.py
python3 analysis_3D.py
python3 analysis_Mall.py
```
