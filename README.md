# Smart Grid Stability — ML Classification Study


## Overview

This project applies supervised machine learning to predict the **stability of a smart electrical grid**. The study follows a structured six-phase methodology covering data analysis, feature engineering, baseline benchmarking, hyperparameter optimisation, statistical validation, model diagnosis, and explainability (XAI).

### Research Framework

![Smart Grid Framework](smart_grid_framework.png)

### Methodology Flowchart

The full step-by-step methodology is documented in [`methodology_flowchart.pdf`](methodology_flowchart.pdf), covering all phases from raw data ingestion through XAI interpretation.

---

## Methodology Phases

| Phase                                                  | Notebooks    | Description                                                                               |
| ------------------------------------------------------ | ------------ | ----------------------------------------------------------------------------------------- |
| **Phase 1 — Data Understanding**                | 01           | EDA, class balance, correlation analysis, train/test split                                |
| **Phase 2 — Feature Engineering**               | 02, 02a, 02b | Feature construction, cross-validated selection, separability analysis, imbalance study   |
| **Phase 3 — Baseline Evaluation**               | 03           | 14-model untuned baseline, scaler sensitivity analysis, optimal scaler per model          |
| **Phase 4 — Hyperparameter Optimisation**       | 04a, 05b     | TPE (Bayesian/Optuna) and Grey Wolf Optimiser (GWO), 50 trials/iterations per model       |
| **Phase 5 — Comparative Analysis & Validation** | 06, 07, 08   | Cross-method comparison, Wilcoxon/Friedman statistical tests, error analysis, calibration |
| **Phase 6 — Explainability (XAI)**              | 09           | SHAP-based global and local explanations for the best model (LightGBM)                    |

---

## Dataset

| Property            | Value                                 |
| ------------------- | ------------------------------------- |
| Source              | UCI Smart Grid Stability (augmented)  |
| Samples             | 60,000                                |
| Raw features        | 12 (tau1–tau4, p1–p4, g1–g4)       |
| Engineered features | 13 selected after feature engineering |
| Target              | Binary — Stable / Unstable           |

**Selected features used in modelling:** `tau1`, `tau2`, `tau3`, `tau4`, `p1`, `g1`, `tau_mean`, `g_sum`, `tau1_g1`, `tau2_g2`, `tau3_g3`, `tau4_g4`, `g_tau_ratio`

---

## Notebooks

| #   | Notebook                                       | Phase   | Description                                                                             |
| --- | ---------------------------------------------- | ------- | --------------------------------------------------------------------------------------- |
| 01  | `01_data_loading_and_eda.ipynb`              | Phase 1 | Data loading, quality checks, EDA, correlation analysis, train/test split               |
| 02  | `02_feature_engineering_and_selection.ipynb` | Phase 2 | Feature construction (ratio/product features) & 5-fold cross-validated selection        |
| 02a | `02a_feature_separability_analysis.ipynb`    | Phase 2 | Linear separability analysis — LDA projections, Fisher's criterion, t-SNE              |
| 02b | `02b_Imbalance_Handling_Study.ipynb`         | Phase 2 | SMOTE variants vs. class-weight: impact on classification performance                   |
| 03  | `03_baseline_evaluation.ipynb`               | Phase 3 | 14-model untuned baseline, per-model optimal scaler selection, test evaluation          |
| 04a | `04a_Bayesian_Optimization_TPE.ipynb`        | Phase 4 | Tree-structured Parzen Estimator (TPE) via Optuna — 50 trials per model                |
| 05b | `05b_Grey_Wolf_Optimization.ipynb`           | Phase 4 | Grey Wolf Optimiser (GWO) — nature-inspired population search, 50 iterations per model |
| 06  | `06_comprehensive_comparison.ipynb`          | Phase 5 | Baseline vs. TPE vs. GWO: accuracy, convergence, runtime comparison                     |
| 07  | `07_statistical_testing.ipynb`               | Phase 5 | Wilcoxon signed-rank & Friedman tests for statistical significance of improvements      |
| 08  | `08_model_diagnosis.ipynb`                   | Phase 5 | Error analysis, probability calibration, generalisation gap assessment                  |
| 09  | `09_XAI_for_LightGBM.ipynb`                  | Phase 6 | SHAP explainability — global summary, dependence plots, waterfall bridge, ICE/PDP      |

---

## Models Evaluated

14 classifiers across linear, kernel, ensemble, and gradient boosting families:

`LR` · `LDA` · `QDA` · `NB` · `KNN` · `LinearSVC` · `SVM` · `AdaBoost` · `RF` · `GB` · `XGBoost` · `LightGBM` · `CatBoost` · `SGD`

---

## Key Results

### Phase 3 — Baseline (Default Hyperparameters, Test Set)

| Model         | Accuracy | F1     | AUC    |
| ------------- | -------- | ------ | ------ |
| CatBoost      | 0.9923   | 0.9939 | 0.9995 |
| XGBoost       | 0.9899   | 0.9921 | 0.9994 |
| SVM           | 0.9823   | 0.9862 | 0.9987 |
| LightGBM      | 0.9711   | 0.9775 | 0.9969 |
| Random Forest | 0.9690   | 0.9759 | 0.9962 |

### Phase 4 — After Hyperparameter Optimisation (Test Set, Top-5)

**TPE (Optuna):**

| Model              | Accuracy         | F1               | AUC              |
| ------------------ | ---------------- | ---------------- | ---------------- |
| **LightGBM** | **0.9991** | **0.9993** | **1.0000** |
| GB                 | 0.9986           | 0.9989           | 1.0000           |
| CatBoost           | 0.9970           | 0.9977           | 1.0000           |
| SVM                | 0.9966           | 0.9973           | 0.9999           |
| XGBoost            | 0.9939           | 0.9953           | 0.9998           |

**GWO (Grey Wolf Optimiser):**

| Model              | Accuracy         | F1               | AUC              |
| ------------------ | ---------------- | ---------------- | ---------------- |
| **LightGBM** | **0.9991** | **0.9993** | **1.0000** |
| CatBoost           | 0.9978           | 0.9983           | 1.0000           |
| GB                 | 0.9972           | 0.9978           | 1.0000           |
| SVM                | 0.9971           | 0.9977           | 0.9998           |
| XGBoost            | 0.9940           | 0.9953           | 0.9998           |

> **LightGBM** (Raw scaler) achieves the highest accuracy under both optimisers and is selected as the target model for XAI analysis.

---

## Optimisation Methods

| Method          | Trials / Iterations | Framework      | Search strategy                         |
| --------------- | ------------------- | -------------- | --------------------------------------- |
| TPE (Bayesian)  | 50 per model        | Optuna         | Probabilistic model-guided search       |
| GWO (Grey Wolf) | 50 per model        | NumPy (custom) | Population-based nature-inspired search |

Both methods use **5-fold stratified cross-validation** on training data as the fitness objective. The optimal scaler determined in Phase 3 is fixed for each model.

---

## Explainability (XAI — Phase 6, Notebook 09)

SHAP TreeExplainer applied to the best model (LightGBM, GWO-optimised, Raw scaler). Four publication-ready IEEE-format figures:

| Figure                          | Description                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------- |
| **SHAP Summary Plot**     | Global feature importance ranked by mean\|SHAP\|; beeswarm distribution          |
| **SHAP Dependence Plots** | Feature-value vs. SHAP effect for top-6 features; coolwarm coloring by SHAP sign |
| **SHAP Waterfall Bridge** | Cumulative contribution chart for a representative stable and unstable case      |
| **ICE / PDP Plots**       | Partial dependence + individual conditional expectation for top-6 features       |

---

## Project Structure

```
smart-grid-stability-v01/
├── data/
│   └── smart_grid_stability_augmented.csv   # Raw dataset (60,000 samples)
├── notebooks/                               # Jupyter notebooks (01–09, phases 1–6)
├── utils/
│   ├── plot_config.py                       # IEEE-style figure configuration
│   └── __init__.py
├── results/
│   ├── figures/                             # Saved PNG/PDF figures (git-ignored)
│   └── tables/                             # CSV/JSON/NPY result artefacts (git-ignored)
├── smart_grid_framework.png                 # Research framework diagram
├── methodology_flowchart.pdf               # Full methodology flowchart
├── .gitignore
└── README.md
```

---

## Environment

| Package      | Version |
| ------------ | ------- |
| Python       | 3.13.2  |
| NumPy        | 2.2.5   |
| pandas       | 2.2.3   |
| scikit-learn | 1.7.1   |
| XGBoost      | 3.0.1   |
| LightGBM     | 4.6.0   |
| CatBoost     | 1.2.8   |
| shap         | latest  |
| optuna       | latest  |

---

## Reproducibility

Run notebooks **in order** (01 → 09). Each notebook saves its outputs to `results/tables/`, which downstream notebooks load as frozen inputs — ensuring full reproducibility with no re-training required in later phases.

```bash
git clone https://github.com/Omar-Zantot/smart-grid-stability-v01.git
cd smart-grid-stability-v01
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap optuna matplotlib seaborn
jupyter notebook
```

---

## Configuration

Plot style and IEEE column widths are centralised in `utils/plot_config.py`:

- IEEE single column: **3.5 in**, double column: **7.16 in**
- Colour-blind-safe palette
- Font: Times New Roman (serif), size 10
- Figure DPI: 150 (display) / 300 (saved)

---

## License

This project is released for academic and research purposes.
