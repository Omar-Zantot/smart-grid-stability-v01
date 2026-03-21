# Smart Grid Stability — ML Classification Study

## Overview

This project applies supervised machine learning to predict the **stability of a smart electrical grid**. The study covers the full pipeline — from exploratory data analysis and feature engineering through baseline evaluation, hyperparameter optimisation with two meta-heuristic methods, statistical significance testing, model diagnosis, and explainability (XAI).

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

| #   | Notebook                                       | Description                                                    |
| --- | ---------------------------------------------- | -------------------------------------------------------------- |
| 01  | `01_data_loading_and_eda.ipynb`              | Data loading, quality checks, EDA, class balance               |
| 02  | `02_feature_engineering_and_selection.ipynb` | Feature construction & cross-validated selection               |
| 02a | `02a_feature_separability_analysis.ipynb`    | Linear separability analysis (LDA, Fisher's criterion)         |
| 02b | `02b_Imbalance_Handling_Study.ipynb`         | SMOTE vs. class-weight comparison                              |
| 03  | `03_baseline_evaluation.ipynb`               | 14-model baseline with scaler sensitivity analysis             |
| 04a | `04a_Bayesian_Optimization_TPE.ipynb`        | TPE (Optuna) hyperparameter optimisation                       |
| 05b | `05b_Grey_Wolf_Optimization.ipynb`           | Grey Wolf Optimiser (GWO) hyperparameter search                |
| 06  | `06_comprehensive_comparison.ipynb`          | Baseline vs. TPE vs. GWO cross-method comparison               |
| 07  | `07_statistical_testing.ipynb`               | Wilcoxon signed-rank & Friedman statistical tests              |
| 08  | `08_model_diagnosis.ipynb`                   | Error analysis, calibration, generalisation gap                |
| 09  | `09_XAI_for_LightGBM.ipynb`                  | SHAP explainability — summary, dependence, waterfall, ICE/PDP |

---

## Models Evaluated

14 classifiers spanning linear, kernel, ensemble, and boosting families:

`LR` · `LDA` · `QDA` · `NB` · `KNN` · `LinearSVC` · `SVM` · `AdaBoost` · `RF` · `GB` · `XGBoost` · `LightGBM` · `CatBoost` · `SGD`

---

## Key Results (Baseline — Test Set)

| Model         | Accuracy | F1     | AUC    |
| ------------- | -------- | ------ | ------ |
| CatBoost      | 0.9923   | 0.9939 | 0.9995 |
| XGBoost       | 0.9899   | 0.9921 | 0.9994 |
| SVM           | 0.9823   | 0.9862 | 0.9987 |
| LightGBM      | 0.9711   | 0.9775 | 0.9969 |
| Random Forest | 0.9690   | 0.9759 | 0.9962 |

The best model after Grey Wolf Optimisation is **LightGBM** (Raw scaler), which serves as the target of the XAI analysis.

---

## Optimisation Methods

| Method          | Trials / Iterations   | Framework |
| --------------- | --------------------- | --------- |
| TPE (Bayesian)  | 50 per model          | Optuna    |
| GWO (Grey Wolf) | Custom implementation | NumPy     |

---

## Explainability (XAI — Notebook 09)

Four publication-ready figures generated for LightGBM:

1. **SHAP Summary Plot** — global feature importance ranked by mean |SHAP|
2. **SHAP Dependence Plots** — individual feature effects for the top-6 features with coolwarm coloring
3. **SHAP Waterfall Bridge** — cumulative contribution chart for a representative stable and unstable case
4. **ICE / PDP Plots** — partial dependence curves with individual conditional expectation profiles

---

## Project Structure

```
smart-grid-stability-v01/
├── data/
│   └── smart_grid_stability_augmented.csv   # Raw dataset
├── notebooks/                               # Jupyter notebooks (01–09)
├── utils/
│   ├── plot_config.py                       # IEEE-style figure configuration
│   └── __init__.py
├── results/
│   ├── figures/                             # Saved PNG/PDF figures (git-ignored)
│   ├── tables/                              # CSV/JSON result artefacts (git-ignored)
│   └── smart_grid_framework.png            # Project framework diagram
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

1. Clone the repository.
2. Install dependencies (see environment table above).
3. Run notebooks **in order** (01 → 09). Each notebook saves its outputs to `results/tables/`, which downstream notebooks load.

```bash
git clone https://github.com/Omar-Zantot/smart-grid-stability-v01.git
cd smart-grid-stability-v01
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap optuna matplotlib
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
