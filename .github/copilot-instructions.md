# Copilot instructions for this repository

## Repository shape

This repo is a collection of independent Python coursework/projects, organized under `projects/` plus archived artifacts in `archive/`. Most work happens in:

- `projects/project-kaggle-logreg-a/` and `projects/project-kaggle-logreg-b/`: logistic-regression pipelines over CSV data in each folder's `data/`.
- `projects/project-numerical-methods-toolbox/`: numerical methods utilities (`models.py`, `metrics.py`) plus standalone experiment scripts (`sales.py`, `satelite.py`, etc.).
- `projects/project-dmd-video-separation/`: Dynamic Mode Decomposition for video processing in `pre.py`.
- `projects/project-svd-experiments/` and `projects/project-logistic-demos/`: smaller standalone demos/experiments.
- `projects/project-quadratic-norm-visualization/`: scripts to visualize quadratic norm contours and eigenvector geometry.
- `projects/project-proyecto-1/`: K-Means clustering implementation and metric comparison analyses in `kmeans/`.
- `archive/bundles/`: preserved zip bundles from legacy submissions.

## Build, test, and lint commands

There is no unified build system, no `pytest` suite, and no configured linter in this repo.

Use script-level execution from each project directory:

```bash
# Install runtime dependencies used across projects
python3 -m pip install -r requirements.txt
```

```bash
# Kaggle2 cross-validation check (closest thing to a repeatable test)
cd projects/project-kaggle-logreg-b && python3 test.py
```

```bash
# Run a single project script (single-check equivalent)
cd projects/project-kaggle-logreg-a && python3 reg.py
```

```bash
# DMD pipeline entrypoint
cd projects/project-dmd-video-separation && python3 pre.py
```

```bash
# Single test (numerical toolbox)
cd projects/project-numerical-methods-toolbox && python3 -m unittest test_models.TestModels.test_generate_matrices_linear
```

```bash
# K-Means analysis run
cd projects/project-proyecto-1/kmeans && python3 analysis_2D.py
```

## High-level architecture

### Kaggle projects (`project-kaggle-logreg-a`, `project-kaggle-logreg-b`)

Both projects follow a simple pipeline split across modules:

1. Data preparation (`datos.py` in Kaggle1, `clean.py` in Kaggle2) loads CSVs and prepares numeric features.
2. Model logic (`log_model.py`/`fun.py`/`model.py`) implements logistic regression from scratch with NumPy.
3. Driver scripts (`reg.py`, `predict.py`, `test.py`) train, evaluate (F1), and write `predicciones.csv`.

Key cross-file behavior: preprocessing and feature ordering are tightly coupled to training; changing column handling in `clean.py`/`datos.py` affects every downstream script.

### Numerical methods (`project-numerical-methods-toolbox/`)

- `models.py` is the shared core (least squares, gradient-based methods, Gauss-Newton/Newton helpers).
- `metrics.py` provides regression error and conditioning metrics.
- Other `.py` files are runnable experiments that import these utilities and print/plot results.

### DMD video workflow (`project-dmd-video-separation/`)

`pre.py` contains the end-to-end primitives: video-to-matrix conversion, DMD decomposition, background/foreground split, frame visualization, and AVI export.

## Key conventions in this codebase

- **Run scripts from their own folder**: many files use relative paths like `data/train_df.csv`; running from repo root often breaks file loading.
- **Module side effects are common**: several files execute training/evaluation at import time (no `if __name__ == "__main__"` guard). Import only utility modules when reusing code.
- **Manual NumPy ML implementations**: logistic regression, metrics, folds, and normalization are implemented directly rather than through sklearn estimators.
- **Bias term convention**: when used, bias is prepended as a leading column of ones (`clean.add_bias`), so weight vectors assume intercept at index `0`.
- **Mixed Spanish/English naming**: variable names, comments, and outputs are bilingual (for example `paciente_id`, `predicciones.csv`, `mejores_transformaciones`); preserve existing naming within each module.
