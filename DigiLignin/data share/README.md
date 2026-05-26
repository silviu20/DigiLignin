# Reproducibility package — *Glass transition temperature prediction in lignin polyurethanes using machine learning on small experimental dataset*

This folder contains the code, dataset, trained model artifacts, and key result
tables required to reproduce every figure and metric reported in the published
article:

> Acaru S. F., Comí M., Falireas P., Vanpoucke D. E. P., Vendamme R.,
> Bernaerts K. V. (2026).
> **Glass transition temperature prediction in lignin polyurethanes using
> machine learning on small experimental dataset.**
> *Materials & Design.* Open access.
> DOI: [10.1016/j.matdes.2026.116265](https://doi.org/10.1016/j.matdes.2026.116265) ·
> [ScienceDirect (PII S0264127526008385)](https://www.sciencedirect.com/science/article/pii/S0264127526008385)

> **Status:** companion to the published article. The article's *Data
> Availability Statement* reads *“The data that support the findings of this
> study are available in the supplementary material of this article.”* This
> folder provides an expanded replication package — the same experimental
> dataset together with the full code, trained model artefacts, and key result
> tables — so that every figure and metric in the article can be regenerated
> end-to-end.

---

## 1. Package layout

```
data share/
├── README.md                  ← this file
├── requirements.txt           ← exact Python dependencies
├── LICENSE                    ← code: MIT, data: CC-BY-4.0
├── CITATION.cff               ← machine-readable citation
├── _build_package.ps1         ← script used to assemble this folder
│
├── data/                      ← raw, unprocessed experimental dataset
│   ├── dataset.csv.xlsx       ← original spreadsheet (full feature set)
│   ├── dataset.xlsx           ← cleaned spreadsheet consumed by training scripts
│   └── README.md              ← column dictionary
│
├── code/                      ← all analysis & plotting scripts, mirroring
│   │                            the manuscript's pipeline order
│   ├── 1.Loading and Preprocessing/
│   ├── 2.Correlation/
│   ├── 3.PCA/
│   ├── 3.Partial dependence plots/
│   ├── 4.Wrapper/             ← feature-selection wrapper experiments
│   ├── 5.Model/               ← stacked-ensemble training
│   ├── 6.Model metrics/       ← performance plots & tables
│   ├── 7.Mapping/             ← Tg surface mapping with the best model
│   ├── 8.Extrapolation/       ← adaptive-grid extrapolation analysis
│   ├── 9.Parallel coordinates plot/
│   ├── 10.Dataset Distribution based on swelling ratio/
│   ├── Universality/          ← applicability domain, permutation, Williams
│   └── Graphical abstract/
│
├── models/best_model/         ← pre-trained best stacked-ensemble (5 features,
│   │                            10 base estimators) — load with joblib
│   ├── best_model_base_models.joblib
│   ├── best_model_meta_model.joblib
│   ├── best_model_x_scaler.joblib
│   ├── best_model_y_scaler.joblib
│   ├── best_model_features.txt
│   └── best_model_metadata.json
│
├── results/                   ← key numerical outputs underlying the figures
│   │                            (top-feature combinations, n-estimator sweep,
│   │                            stratified-split results, mapping summary,
│   │                            applicability-domain & permutation tables, …)
│   └── ...
│
└── figures/                   ← the eleven main-text figures of the article
    │                            (`Fig_01…Fig_11`, PNG, extracted from the
    │                            published manuscript) plus the graphical
    │                            abstract.
```

The large derived artefacts that can be regenerated (≈250 MB full mapping CSV,
hundreds of per-combination scaler `.joblib` files from the wrapper run) are
**not** shipped here. The scripts that produce them are provided and the
expected runtime is documented in
[code/7.Mapping/README_BEST_MODEL_MAPPING.md](code/7.Mapping/README_BEST_MODEL_MAPPING.md).

---

## 2. Environment

* Operating system: tested on Windows 10/11 (paths use `\`), but pure-Python
  scripts run unchanged on Linux/macOS.
* Python: **3.10 or 3.11** recommended.

Create and populate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell
# .venv\Scripts\activate.bat          # cmd
# source .venv/bin/activate           # bash
pip install -r requirements.txt
```

The full dependency list is in [requirements.txt](requirements.txt). All
package versions are pinned to those used to produce the published results so
that the random seeds yield bit-identical outputs.

---

## 3. Reproducing the published results

The numerical pipeline is deterministic (`RANDOM_SEED = 42`, `n_jobs = 1`
inside grid searches that affect reported metrics). Each step writes its own
outputs; later steps can be skipped if you only want figures from the already-
trained best model in `models/best_model/`.

| # | Step | Script (run from `code/<folder>`) | Outputs |
|---|---|---|---|
| 1 | Load & preprocess | `1.Loading and Preprocessing/Loading and preprocessing.py` | scaled DataFrame in memory |
| 2 | Correlation heat-map (**Fig. 3**) | `2.Correlation/Correlation plot.py` | PNG |
| 3 | Partial-dependence plots (**Fig. 4**) | `3.Partial dependence plots/Code_R_Sr and Tg1.py`, `Code_R_Sr and Tg_Co-polyol and Ratio.py`, `Merging partial dependece plots.py` | PNGs |
| 4 | PCA loadings (**Fig. 5**) | `3.PCA/a.Data Analysis_PCA.py`, `b.Cumulative variance of the PCs1.py`, `c.PCA plot.py` | PNGs |
| 5 | Feature-selection wrapper (Supporting Info. §3; Table 4 candidates) | `4.Wrapper/Testing_feature_combinations.py` then `Plot_feat_comb_MAE.py` | CSVs + PNGs |
| 6 | Stratified split + top-models table (**Table 4**) | `4.Wrapper/Stratified_fixed_split_16_val_16_test/` scripts | `fixed_split_results.csv`, `top_10_models_table.csv` |
| 7 | n-estimators sweep (**Fig. 6**) | `4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/` scripts → `code/6.Model metrics/plot_model6_n_estimators_performance.py` | CSV + PNG |
| 8 | Final stacked-ensemble training (pipeline shown in **Fig. 2**) | `5.Model/Stacked_Ensembles_Fixed.py` and `7.Mapping/retrain_best_model.py` | metrics + joblib artefacts in `models/best_model/` |
| 9 | Individual-vs-ensemble metrics (**Table 5**) | `6.Model metrics/Stacked_ensemble_performance_with comments.py` | `individual_models_performance.csv` |
| 10 | Predicted-vs-actual & residual plots (**Fig. 7**) | `6.Model metrics/plot_ensemble_predicted_vs_actual.py`, `scatter plot predicted_vs_actual.py` | PNGs |
| 11 | Tg surface mapping (**Fig. 8**) | `7.Mapping/mapping_best_model.py` (or `…_fast.py`) → `Density plot for mapping data.py`, `Distribution of Predicted Tg Values_mapped_results.py` | ~250 MB CSV + density PNGs |
| 12 | Applicability domain & permutation test (**Fig. 9**, Fig. S7) | `Universality/` scripts | `ad_summary.csv`, `permutation_results.csv`, `williams_plot.png` |
| 13 | Extrapolation analysis (**Fig. 10**) | `8.Extrapolation/adaptive_grid_search_best_model.py` then `extrapolation_plot_best_model_labeled.py` | CSV + PNGs |
| 14 | Parallel-coordinates UI (**Fig. 11**) | `9.Parallel coordinates plot/parallel_coordinates_best_model.py` | interactive HTML |

Several scripts contain absolute paths (e.g. `C:/Users/.../dataset/...`) from
the original development machine. Replace those strings with a path that
points to `data/dataset.csv.xlsx` or `data/dataset.xlsx` inside this package
before running.

---

## 4. Loading the trained best model

The model published in the manuscript can be loaded directly — no retraining
required:

```python
import json, joblib, numpy as np, pandas as pd
from pathlib import Path

MODEL_DIR = Path("models/best_model")

base_models = joblib.load(MODEL_DIR / "best_model_base_models.joblib")
meta_model  = joblib.load(MODEL_DIR / "best_model_meta_model.joblib")
x_scaler    = joblib.load(MODEL_DIR / "best_model_x_scaler.joblib")
y_scaler    = joblib.load(MODEL_DIR / "best_model_y_scaler.joblib")
features    = (MODEL_DIR / "best_model_features.txt").read_text().splitlines()
meta        = json.loads((MODEL_DIR / "best_model_metadata.json").read_text())

# features = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r',
#             'Copolyol (wt%)', 'Isocyanate (wt%)']

def predict_tg(X):
    X = pd.DataFrame(X, columns=features)
    Xs = x_scaler.transform(X)
    meta_feats = np.column_stack([m.predict(Xs) for m in base_models])
    y_scaled = meta_model.predict(meta_feats).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled).ravel()

# Example
print(predict_tg([[35, 250, 1.0, 30, 12]]))
```

Reported metrics (see `best_model_metadata.json`):

| Set | R² | MSE | MAE |
|---|---|---|---|
| Train (104 samples) | 0.942 | 41.94 | 3.92 |
| Validation (16) | 0.494 | 391.95 | 13.41 |
| Test (16) | 0.547 | 338.37 | 15.17 |

These are the **Rank 1** entry of Table 4 and the **Stacking Ensemble** row of
Table 5 in the published article. The same numbers are stored verbatim in
`results/6.Model metrics/Stratified Stacked Ensemble/individual_models_performance.csv`
(row `n_estimators=10`, `Model=Ensemble`).

---

## 5. Mapping the figures and tables of the article to this package

Every main-text figure in the article is shipped under `figures/` with the
same numbering (`Fig_01…Fig_11`, PNG extracted from the published manuscript).
Numerical artefacts that back the figures and tables live under `results/`:

| Article element | File in this package |
|---|---|
| Figure 6 (n-estimators sweep) | `results/4.Wrapper/Fixed_stacking_ensemble_with_n_estimators/all_combinations_n_estimators_results.csv`, `model6_n_estimators_results.csv` |
| Figure 7 (Pearson r = 0.73, combined MAE = 14.29 °C) | regenerated from `models/best_model/` via `code/6.Model metrics/plot_ensemble_predicted_vs_actual.py` |
| Figure 8 (KDE mapping, >4 M combinations) | `results/7.Mapping/mapping_summary.json` reports `total_combinations = 4 976 580`; densities regenerated from `mapped_results_tg_best_model.csv` (not shipped — ~250 MB; reproducible) |
| Figure 9B (AD coverage 98.1 % / 68.8 % / 62.5 %) | `results/Universality/ad_summary.csv` |
| Figure 10 (extrapolation: lower plateau 5.6 °C, upper plateau 79.4 °C) | `results/8.Extrapolation/closest_inputs_best_model.csv` |
| Permutation test (p < 0.001 vs MAE 13.41 °C) | `results/Universality/permutation_results.csv` (1000 shuffled MAEs) |
| Table 4 (top-5 wrapper combinations) | `results/4.Wrapper/Stratified_fixed_split_16_val_16_test/top_10_models_table.csv` + `fixed_split_results.csv`; the older OOF-based scan (referenced in Supporting Information §3.2) is in `results/4.Wrapper/top_5_*` |
| Table 5 (individual vs ensemble) | `results/6.Model metrics/Stratified Stacked Ensemble/individual_models_performance.csv` |

> **Note on Wrapper results.** Two independent feature-selection runs are
> archived: an Out-of-Fold cross-validation scan (top-level files in
> `results/4.Wrapper/`) and a stratified fixed-split scan
> (`Stratified_fixed_split_16_val_16_test/`). The published main text (Table 4
> and Table 5) reports the stratified split; Supporting Information §3.2
> reports the OOF comparison (best OOF MAE = 15.71 °C vs stratified
> 13.41 °C). Both result sets are kept in the package for full traceability.

---

## 6. Dataset

Sample-level experimental measurements of lignin-based polyurethane
formulations and their thermo-mechanical properties.

* `data/dataset.csv.xlsx` — original spreadsheet with all measured columns.
* `data/dataset.xlsx` — cleaned spreadsheet (Tg column renamed `Tg(deg C)`)
  used directly by the training scripts in `code/7.Mapping/` and
  `code/4.Wrapper/Stratified_fixed_split_16_val_16_test/`.
* `data/README.md` — column dictionary, units, missing-value convention.

---

## 7. License & citation

* **Code** (everything under `code/`, `_build_package.ps1`): MIT License.
* **Data** (`data/`) and figures: Creative Commons Attribution 4.0
  International (CC-BY-4.0).

See [LICENSE](LICENSE) for full text. If you use this package, please cite the
published article:

> Acaru, S. F.; Comí, M.; Falireas, P.; Vanpoucke, D. E. P.; Vendamme, R.;
> Bernaerts, K. V. *Glass transition temperature prediction in lignin
> polyurethanes using machine learning on small experimental dataset.*
> Materials & Design, 2026. doi:[10.1016/j.matdes.2026.116265](https://doi.org/10.1016/j.matdes.2026.116265).

Machine-readable metadata: [CITATION.cff](CITATION.cff).

---

## 8. Contact

For questions contact the
corresponding author of the manuscript.
