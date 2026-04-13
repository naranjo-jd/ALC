# Portfolio overview

## What is currently in this repository

| Project | Focus | Main techniques | Output style |
| --- | --- | --- | --- |
| `project-numerical-methods-toolbox` | Core numerical computation | least squares, LU/QR, gradient descent, Newton/Gauss-Newton | scripts + notebook |
| `project-dmd-video-separation` | Matrix methods for dynamics | DMD, low-rank/sparse separation | generated videos + notebook |
| `project-kaggle-logreg-a` | Binary classification | custom logistic regression with preprocessing | CSV predictions + script |
| `project-kaggle-logreg-b` | Model evaluation and transforms | k-fold CV, F1 scoring, feature transforms | metrics printout + CSV |
| `project-svd-experiments` | Matrix decomposition intuition | SVD construction and geometric visualization | plots/scripts |
| `project-logistic-demos` | Compact method demos | logistic training + decision boundary plotting | scripts/notebook |
| `project-quadratic-norm-visualization` | Geometry of induced norms | quadratic form norm contours, eigendecomposition interpretation | scripts + plots |
| `project-proyecto-1` | Unsupervised clustering | custom K-Means, Euclidean/Manhattan/Chebyshev/Mahalanobis metrics, inertia comparison by seed | scripts + notebooks + plots |
| `project-proyecto-2-analysis` | exploratory analysis | notebook and report-style analysis | HTML/PDF/notebook |

## Suggested portfolio additions (high impact)

1. Add a short methods note per flagship project (problem, model equations, algorithm, complexity, limits).
2. Add reproducible benchmark scripts comparing custom implementations vs NumPy/SciPy/sklearn baselines.
3. Add lightweight tests for `project-numerical-methods-toolbox/models.py` and `metrics.py`.
4. Add deterministic output folders (`outputs/`) with one curated artifact per project and scripts to regenerate them.
5. Add one polished end-to-end narrative notebook linking linear algebra theory to observed empirical behavior.
