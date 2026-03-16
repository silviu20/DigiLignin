# -*- coding: utf-8 -*-
"""
Publication-quality diagram of the Stacked Ensemble development and
unseen-data inference pipeline for DigiLignin Tg prediction.

Outputs: stacking_ensemble_diagram.png  (600 dpi, journal-ready)
         stacking_ensemble_diagram.svg  (vector, scalable)
         stacking_ensemble_diagram.pdf  (vector, for LaTeX)

Run with:
    C:\\Users\\sacaru\\AppData\\Local\\miniconda3\\python.exe stacking_ensemble_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import os

# ── Colour palette (colourblind-friendly, prints well in greyscale) ────────
C = {
    "data":   "#2166ac",   # dark blue      – raw dataset
    "prep":   "#4dac26",   # green          – data prep / features
    "cv":     "#7b3294",   # purple         – cross-validation
    "scale":  "#d7191c",   # red            – scaling
    "base":   "#1a7837",   # forest green   – base models
    "oof":    "#e08400",   # amber          – OOF predictions
    "meta":   "#c51b7d",   # magenta        – meta-model
    "eval":   "#2980b9",   # steel blue     – evaluation
    "sweep":  "#b5390a",   # brick red      – n_estimators sweep
    "final":  "#1a5276",   # navy           – final training
    "unseen": "#0e6655",   # teal           – inference
    "out":    "#1b5e20",   # deep green     – output
    "arrow":  "#2c2c2c",
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    7.5,
    "svg.fonttype": "none",   # editable text in Inkscape / Illustrator
})

# ── Canvas ──────────────────────────────────────────────────────────────────
CM = 1 / 2.54
fig = plt.figure(figsize=(19 * CM, 36 * CM))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")

# layout anchors
LM = 0.060    # left content margin (right of section labels)
RM = 0.982    # right content margin
CX = (LM + RM) / 2


# ═══════════════════════════════════════════════════════════════════════════
# DRAWING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

def rbox(x, y, w, h, text, color, fs=7.2, bold=False,
         alpha=0.11, lw=1.2, r=0.009, tc=None):
    """Rounded rectangle with centred multi-line text."""
    tc = tc or color
    for fc, a, zo in [(color, alpha, 3), ("none", 1, 4)]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.003,rounding_size={r}",
            linewidth=lw, edgecolor=color, facecolor=fc, alpha=a, zorder=zo))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.45, zorder=5)


def db_shape(x, y, w, h, text, color, fs=7.2):
    """Cylinder (dataset) shape."""
    ey = h * 0.22
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-w/2, y-h/2+ey/2), w, h-ey,
        boxstyle="square,pad=0", linewidth=1.2,
        edgecolor=color, facecolor=color, alpha=0.11, zorder=3))
    for cy, a in [(y+h/2-ey/2, 0.28), (y-h/2+ey/2, 0.11)]:
        ax.add_patch(mpatches.Ellipse(
            (x, cy), w, ey,
            facecolor=color, alpha=a, edgecolor=color, linewidth=1.2, zorder=4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold", linespacing=1.45, zorder=5)


def diamond(x, y, w, h, text, color, fs=7.2):
    """Diamond decision node."""
    pts = [[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]]
    ax.add_patch(mpatches.Polygon(
        pts, closed=True, facecolor=color, alpha=0.13,
        edgecolor=color, linewidth=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold", linespacing=1.45, zorder=5)


def arr(x1, y1, x2, y2, color=None, lw=1.1):
    """Downward arrow between two points."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle="arc3,rad=0",
        arrowstyle="->, head_length=0.012, head_width=0.006",
        color=color or C["arrow"], linewidth=lw, zorder=6))


def section_bg(y_top, y_bot, label, color):
    """Shaded band with rotated label on the left margin."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (LM - 0.003, y_bot), RM - LM + 0.003, y_top - y_bot,
        boxstyle="round,pad=0.003,rounding_size=0.006",
        linewidth=0.7, edgecolor=color, facecolor=color,
        alpha=0.06, zorder=1))
    ax.text(LM - 0.033, (y_top + y_bot) / 2, label,
            fontsize=6.5, color=color, fontweight="bold",
            va="center", ha="center", rotation=90, zorder=2)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BANDS
# ═══════════════════════════════════════════════════════════════════════════
section_bg(0.993, 0.862, "DATA\nPREPARATION",              C["prep"])
section_bg(0.855, 0.370, "MODEL DEVELOPMENT\n— NESTED CV",  C["cv"])
section_bg(0.363, 0.193, "n_estimators SWEEP\n& SELECTION", C["sweep"])
section_bg(0.186, 0.048, "UNSEEN DATA\nINFERENCE",          C["unseen"])


# ═══════════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ═══════════════════════════════════════════════════════════════════════════
ax.text(CX, 0.981,
        "Stacked Ensemble Pipeline for Tg Prediction of Lignin-based Polyurethane Foams",
        ha="center", va="center", fontsize=10.0, fontweight="bold",
        color="#111111", zorder=10)
ax.text(CX, 0.969,
        "Rank-1 Feature Combination  ·  5 Level-0 Base Models  ·  Ridge Level-1 Meta-Model  "
        "·  Out-of-Fold Stacking  ·  RepeatedKFold (5 × 2)",
        ha="center", va="center", fontsize=7.0, color="#444444", zorder=10)


# ═══════════════════════════════════════════════════════════════════════════
# A — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════

# A1  Raw dataset
db_shape(CX, 0.935, 0.38, 0.046,
         "Raw Dataset  (dataset.xlsx)\n"
         "Target: Tg (°C)  ·  Lignin-based polyurethane foam experiments",
         C["data"], fs=7.2)

arr(CX, 0.912, CX, 0.900)

# A2  Preprocessing
rbox(CX, 0.889, 0.80, 0.022,
     "Remove rows with NaN Tg (°C)  ·  "
     "Encode 'Isocyonate type': N3600 → 1, HDI → 0  ·  Fill residual NaN → 0",
     C["prep"], fs=7.0)

arr(CX, 0.878, CX, 0.866)

# A3  Rank-1 feature combination
rbox(CX, 0.855, 0.84, 0.022,
     "Rank-1 Feature Combination  (5 input features)\n"
     "①  Lignin (wt%)    ②  Co-polyol type (PTHF)    ③  r (Copolyol ratio)    "
     "④  Copolyol (wt%)    ⑤  Sratio (%)",
     C["prep"], fs=7.4, bold=True)

arr(CX, 0.844, CX, 0.832)


# ═══════════════════════════════════════════════════════════════════════════
# B — MODEL DEVELOPMENT  (nested CV)
# ═══════════════════════════════════════════════════════════════════════════

# B1  Outer CV
rbox(CX, 0.820, 0.66, 0.022,
     "Outer CV:  RepeatedKFold  (n_splits=5, n_repeats=2, random_state=42)  "
     "→  10 outer folds   [X_train_outer  |  X_val_outer]",
     C["cv"], fs=7.0)

arr(CX, 0.809, CX, 0.797)

# B2  RobustScaler
rbox(CX, 0.786, 0.68, 0.022,
     "RobustScaler  fit on X_train_outer only  →  transform X_val_outer"
     "  (separate scaler fitted on y_train_outer)",
     C["scale"], fs=7.0)

arr(CX, 0.775, CX, 0.763)

# B3  Level-0 header
rbox(CX, 0.753, 0.74, 0.020,
     "Level-0  Base Models — hyperparameter tuning via GridSearchCV  "
     "(inner 5-fold CV,  scoring = neg_MSE)",
     C["base"], fs=7.2, bold=True)

# B4  Five base model boxes (fan out)
BM_Y = 0.686
BM_H = 0.082
BM_W = 0.164
GAP  = (RM - LM - 5 * BM_W) / 4
BM_XS = [LM + BM_W/2 + i * (BM_W + GAP) for i in range(5)]

BM_TEXTS = [
    "Gradient Boosting\nRegressor\n"
    "─────────────\nn_estimators: swept\n"
    "learning_rate:\n{0.01, 0.1, 0.2}\n"
    "max_depth: {3, 5, 7}",

    "Random Forest\nRegressor\n"
    "─────────────\nn_estimators: swept\n"
    "max_depth:\n{None, 10, 20}\n"
    "min_samples_split:\n{2, 5, 10}",

    "Support Vector\nRegressor (SVR)\n"
    "─────────────\nC: {0.1, 1, 10}\n"
    "kernel:\n{rbf, linear}\n"
    "gamma: {scale, auto}",

    "Lasso\n(L1 Regression)\n"
    "─────────────\nalpha:\n{0.1, 1, 10}\n"
    "max_iter:\n{1000, 5000}",

    "ElasticNet\n(L1 + L2)\n"
    "─────────────\nalpha: {0.1, 1, 10}\n"
    "l1_ratio:\n{0.1, 0.5, 0.9}\n"
    "max_iter:\n{1000, 5000}",
]

for bx, bt in zip(BM_XS, BM_TEXTS):
    arr(CX, 0.743, bx, BM_Y + BM_H/2, color=C["base"], lw=0.8)
    rbox(bx, BM_Y, BM_W, BM_H, bt, C["base"], fs=6.5)

# B5  OOF predictions
OOF_Y = 0.565
for bx in BM_XS:
    arr(bx, BM_Y - BM_H/2, CX, OOF_Y + 0.022, color=C["oof"], lw=0.8)

rbox(CX, OOF_Y, 0.82, 0.030,
     "Out-of-Fold (OOF) Predictions  via  cross_val_predict  (inner 5-fold CV)\n"
     "→  meta-feature matrix  [n_train_outer × 5]  ·  "
     "each sample predicted only when held out  ·  data leakage prevented",
     C["oof"], fs=7.2, bold=True)

arr(CX, 0.550, CX, 0.538)

# B6  Retrain base models + validation meta-features
rbox(CX, 0.527, 0.80, 0.022,
     "Retrain each best base model on full X_train_outer  →  predict X_val_outer  "
     "→  validation meta-feature matrix  [n_val × 5]",
     C["base"], fs=7.0)

arr(CX, 0.516, CX, 0.504)

# B7  Ridge meta-model
rbox(CX, 0.493, 0.64, 0.022,
     "Level-1 Meta-Model:  Ridge Regression  (random_state = 42)\n"
     "Fit on OOF meta-features  [n_train_outer × 5]  →  predict on validation meta-features",
     C["meta"], fs=7.2, bold=True)

arr(CX, 0.482, CX, 0.470)

# B8  Evaluation
rbox(CX, 0.459, 0.72, 0.022,
     "Evaluate on X_val_outer  ·  inverse-transform via y_scaler\n"
     "Per-fold metrics:  R²  ·  MSE  ·  MAE   (training set  +  validation set)",
     C["eval"], fs=7.0)

arr(CX, 0.448, CX, 0.436)

# B9  Aggregation
rbox(CX, 0.425, 0.64, 0.022,
     "Aggregate across 10 outer folds:  Mean ± 95% Confidence Interval\n"
     "(Student t-distribution  via  scipy.stats.sem  and  t.ppf)",
     C["cv"], fs=7.0)

# B10  Dashed loop: repeat for each outer fold
LOOP_X = RM - 0.002
ax.plot([CX + 0.005, LOOP_X], [0.435, 0.435], color=C["cv"], lw=0.9, ls="--", zorder=6)
ax.plot([LOOP_X, LOOP_X], [0.435, 0.830], color=C["cv"], lw=0.9, ls="--", zorder=6)
ax.annotate("", xy=(CX + 0.005, 0.830), xytext=(LOOP_X, 0.830),
            arrowprops=dict(arrowstyle="->", color=C["cv"], lw=0.9))
ax.text(LOOP_X + 0.003, (0.435 + 0.830)/2,
        "Repeat\nfor each\nouter fold",
        fontsize=6.0, color=C["cv"], ha="left", va="center", style="italic", zorder=7)

arr(CX, 0.414, CX, 0.400)

# B11  Note: loop runs for all feature combinations
ax.text(CX, 0.392,
        "The above nested-CV loop is executed for every feature combination "
        "(mandatory features + optional subsets)",
        ha="center", va="center", fontsize=6.5, color=C["cv"],
        style="italic", zorder=7)

arr(CX, 0.384, CX, 0.372)

# B12  Results table
rbox(CX, 0.361, 0.76, 0.022,
     "Results table:  mean R²  ·  MSE  ·  MAE  ± 95% CI  "
     "per  (feature combination × n_estimators value)",
     C["cv"], fs=7.0)


# ═══════════════════════════════════════════════════════════════════════════
# C — N_ESTIMATORS SWEEP & MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════

arr(CX, 0.350, CX, 0.338)

# C1  Sweep box
rbox(CX, 0.326, 0.82, 0.024,
     "n_estimators Sweep  applied to GradientBoosting & RandomForest:\n"
     "{ 1 · 10 · 50 · 100 · 200 · 300 · 400 · 500 · 600 · 700 · 800 · 900 · 1000 }",
     C["sweep"], fs=7.2)

arr(CX, 0.314, CX, 0.300)

# C2  Decision diamond
diamond(CX, 0.286, 0.56, 0.030,
        "Select optimal n_estimators\n(min validation MAE  /  max validation R²)",
        C["sweep"], fs=7.2)

arr(CX, 0.271, CX, 0.259)

# C3  Final training
rbox(CX, 0.247, 0.76, 0.024,
     "Final Model Training on FULL dataset  (selected optimal n_estimators)\n"
     "RobustScaler refit on all data  ·  OOF meta-model retrained on complete set",
     C["final"], fs=7.2, bold=True)

arr(CX, 0.235, CX, 0.223)

# C4  Save artefacts
rbox(CX, 0.211, 0.80, 0.024,
     "Artefacts saved to disk:\n"
     "base_models_fixed_run_1_*.joblib  ·  meta_model_fixed_run_1_*.joblib  "
     "·  x_scaler.joblib  ·  y_scaler.joblib",
     C["final"], fs=7.0)


# ═══════════════════════════════════════════════════════════════════════════
# D — UNSEEN DATA INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

arr(CX, 0.199, CX, 0.187)

# D1  Unseen input
rbox(CX, 0.175, 0.58, 0.024,
     "New / Unseen Sample  (5 features required)\n"
     "①  Lignin (wt%)    ②  Co-polyol type (PTHF)    ③  r    "
     "④  Copolyol (wt%)    ⑤  Sratio (%)",
     C["unseen"], fs=7.2, bold=True)

arr(CX, 0.163, CX, 0.152)

# D2  Three horizontal inference steps
INF_W  = 0.240
INF_H  = 0.030
INF_Y  = 0.139
GAP_I  = (RM - LM - 3 * INF_W) / 2
INF_XS = [LM + INF_W/2 + i * (INF_W + GAP_I) for i in range(3)]

INF_TEXTS = [
    "Apply saved RobustScaler\n(transform only — no refit)",
    "5 Base Models predict\n→ meta-feature vector  [1 × 5]",
    "Ridge Meta-Model predicts\n→ inverse-transform via y_scaler",
]
INF_COLS = [C["scale"], C["base"], C["meta"]]

for ix, it, ic in zip(INF_XS, INF_TEXTS, INF_COLS):
    rbox(ix, INF_Y, INF_W, INF_H, it, ic, fs=7.0)

arr(INF_XS[0]+INF_W/2, INF_Y, INF_XS[1]-INF_W/2, INF_Y)
arr(INF_XS[1]+INF_W/2, INF_Y, INF_XS[2]-INF_W/2, INF_Y)

# fan-in arrow from D1 down to first inference box
arr(CX, 0.152, INF_XS[0], INF_Y + INF_H/2)

# fan-out arrow from last inference box to output
arr(INF_XS[2], INF_Y - INF_H/2, CX, 0.108)

# D3  Output node
rbox(CX, 0.095, 0.50, 0.026,
     "Predicted  Tg (°C)\nfor the New Lignin-based Polyurethane Foam Sample",
     C["out"], fs=9.0, bold=True, alpha=0.20, lw=1.8)


# ═══════════════════════════════════════════════════════════════════════════
# LEGEND  (two rows, below output node)
# ═══════════════════════════════════════════════════════════════════════════
LEG_ITEMS = [
    (C["data"],   "Raw dataset"),
    (C["prep"],   "Data preparation / features"),
    (C["cv"],     "Cross-validation / aggregation"),
    (C["scale"],  "Scaling (RobustScaler)"),
    (C["base"],   "Level-0 base models"),
    (C["oof"],    "OOF meta-features"),
    (C["meta"],   "Level-1 meta-model (Ridge)"),
    (C["eval"],   "Evaluation metrics"),
    (C["sweep"],  "n_estimators sweep / selection"),
    (C["final"],  "Final training / saved artefacts"),
    (C["unseen"], "Unseen-data inference"),
    (C["out"],    "Prediction output"),
]

LEG_TOP = 0.066
SQ      = 0.010
TDX     = 0.013
COLS    = 6
COL_W   = (RM - LM) / COLS
ROW_H   = 0.018

ax.text(LM, LEG_TOP + 0.010, "Legend:",
        fontsize=7.0, fontweight="bold", color="#1a1a1a",
        va="center", ha="left", zorder=9)

for i, (col, lbl) in enumerate(LEG_ITEMS):
    r, c = divmod(i, COLS)
    lx = LM + c * COL_W
    ly = LEG_TOP - r * ROW_H
    ax.add_patch(mpatches.Rectangle(
        (lx, ly - SQ/2), SQ, SQ,
        facecolor=col, alpha=0.40, edgecolor=col, linewidth=0.7, zorder=8))
    ax.text(lx + TDX, ly, lbl,
            fontsize=6.1, va="center", color="#222222", zorder=9)

# figure label
ax.text(LM, 0.991, "Figure 1.",
        fontsize=8.5, fontweight="bold",
        va="top", ha="left", color="#111111", zorder=10)


# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

for fmt, dpi in [("png", 600), ("svg", None), ("pdf", None)]:
    path = os.path.join(OUT_DIR, f"stacking_ensemble_diagram.{fmt}")
    kw = dict(bbox_inches="tight", facecolor="white")
    if dpi:
        kw["dpi"] = dpi
    fig.savefig(path, format=fmt, **kw)
    print(f"Saved → {path}")

plt.close(fig)
print("Done.")
