# Dataset — column dictionary

Two spreadsheets are provided:

* **`dataset.csv.xlsx`** — original export containing the formulation
  features (lignin, co-polyol, isocyanate, ratio, catalyst, …) together with
  the measured glass transition temperature (`Tg`) and swelling ratio
  (`Sratio`). Used by the preprocessing pipeline in
  `code/1.Loading and Preprocessing/`.
* **`dataset.xlsx`** — cleaned working sheet (target column renamed to
  `Tg(deg C)`) consumed directly by `code/7.Mapping/retrain_best_model.py`
  and `code/4.Wrapper/Stratified_fixed_split_16_val_16_test/`.

## Sample count

* **180** lignin-based polyurethane films were synthesised and characterised.
* **136** samples remain after the preprocessing described in the article
  (rows with missing `Tg` and rows with 0 wt% lignin are removed).
* The 136-sample working set is further partitioned with the stratified
  fixed-split routine into **104 training / 16 validation / 16 test** samples.

## Columns (cleaned sheet)

The nine input features and the target follow the manuscript's Table 1
(features marked `*` are the *mandatory* set used by the wrapper feature
selection; the remaining features are *optional* and combined exhaustively).

| Column in spreadsheet | Manuscript name | Unit | Mandatory | Description |
|---|---|---|---|---|
| `Sample name` | — | — | — | Internal sample identifier |
| `Lignin (wt%)` | Lignin (wt%) | wt % | * | Lignin content in the formulation |
| `Copolyol (wt%)` | Co-polyol (wt%) | wt % |  | Co-polyol content |
| `Co-polyol type (PTHF)` | Co-polyol type (PTHF) | g/mol | * | Molecular weight of the PTHF co-polyol (250, 650 or 1000) |
| `Isocyanate (wt%)` | Isocyanate (wt%) | wt % |  | Isocyanate content |
| `Isocyanate (mmol NCO)` | Isocyanate (mmol NCO) | mmol/g |  | NCO equivalents per gram of formulation |
| `Isocyonate type` *(sic)* | Isocyanate type | categorical |  | `HDI` → 0, `N3600` → 1 |
| `r` | Ratio | — | * | Mixing molar ratio [NCO]/[OH] |
| `tin(II) octoate` | Tin(II) octoate (wt%) | wt % |  | Catalyst loading |
| `Sratio(%)` | Swelling ratio (%) | % |  | Post-synthesis characterisation — used as an input feature only in some wrapper variants |
| `Tg(deg C)` | Tg (°C) | °C | **target** | Glass transition temperature (DSC, second heating step) |

> The spreadsheet preserves two historical naming quirks from the lab
> notebook: the column for the mixing ratio is named `r` instead of `Ratio`,
> and the isocyanate-type column contains the typo `Isocyonate type`. The
> training and mapping scripts shipped in `code/` reference these exact
> column names, so they are kept verbatim in the data files.

## Conventions

* Missing values are encoded as `"-"` in the raw CSV-style sheet; the
  preprocessing script converts them to `NaN`.
* Categorical `Isocyonate type` is mapped to `{HDI: 0, N3600: 1}` (any other
  value → `NaN` → filled with 0).
* All continuous features are scaled with `sklearn.preprocessing.RobustScaler`
  fit on the training split only.

## Provenance

Lignin-based PU films were synthesised and characterised within the
*digiLignin* project (FWO grant G0E0223N). The spreadsheet provided here is
the same data that underlies every figure and metric in the published
article and the `dataset.csv` referenced in its Supporting Information. See
the article's *Materials and Methods* section for the full synthesis and
characterisation protocols.
