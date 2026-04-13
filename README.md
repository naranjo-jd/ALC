# ALC - Computational Linear Algebra Portfolio

This repository contains portfolio-ready projects from a Computational Linear Algebra course, organized by project and method focus.

## Repository map

- `projects/project-numerical-methods-toolbox/`: reusable numerical methods (least squares, QR/LU helpers, Newton/Gauss-Newton, error metrics).
- `projects/project-dmd-video-separation/`: Dynamic Mode Decomposition for background/foreground video separation.
- `projects/project-kaggle-logreg-a/`: logistic regression pipeline over tabular medical data.
- `projects/project-kaggle-logreg-b/`: extended logistic workflow with CV and feature-transform exploration.
- `projects/project-svd-experiments/`: SVD decomposition and visualization experiments.
- `projects/project-logistic-demos/`: compact logistic-regression demos and plotting.
- `projects/project-quadratic-norm-visualization/`: geometric visualization of quadratic norms and eigenvector directions.
- `projects/project-proyecto-1/`: K-Means clustering project with custom distance metrics and comparative analysis scripts.
- `projects/project-proyecto-2-analysis/`: exploratory analysis/notebook-based project artifacts.
- `archive/`: legacy bundles and preserved submission artifacts.

## Quick start

```bash
python3 -m pip install numpy pandas scipy sympy matplotlib scikit-learn opencv-python pydmd
python3 -m pip install seaborn
```

Run examples from project directories:

```bash
cd projects/project-kaggle-logreg-b && python3 test.py
cd projects/project-kaggle-logreg-a && python3 reg.py
cd projects/project-dmd-video-separation && python3 pre.py
cd projects/project-proyecto-1/kmeans && python3 analysis_2D.py
```

## Portfolio guide

See `PORTFOLIO.md` for:

- what each project demonstrates,
- strongest showcase pieces,
- recommended additions to improve portfolio impact.

See `BIBLIOGRAPHY.md` for reference books/papers used in the course context (tracked as citations rather than committed PDFs).
