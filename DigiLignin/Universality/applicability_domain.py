# -*- coding: utf-8 -*-
"""
Applicability Domain (AD) Analysis — Williams Plot
====================================================
Stacking Ensemble for Tg Prediction in Lignin Polyurethanes

Method:
    Leverage-based Applicability Domain using the HAT (projection) matrix
    computed on the 5-feature scaled training set.

    - Leverage threshold:         h* = 3(k+1)/n
    - Standardized residual limit: |z| <= 3

    Williams plot displays (h_i, z_i) for training, validation and test sets,
    with AD boundaries drawn as dashed lines.

    Additional plots:
        - AD coverage bar chart
        - Feature coverage heatmap (normalised feature ranges vs. training)
        - Permutation test on validation MAE

Data source:
    4.Wrapper/Fixed_Stacking_Ensemble/dataset.xlsx
    7.Mapping/  (saved model artifacts: joblib + metadata)

Output (saved to Universality/):
    williams_plot.pdf/.png/.svg
    ad_coverage.pdf/.png/.svg
    feature_coverage.pdf/.png/.svg
    permutation_test.pdf/.png/.svg
    ad_summary.csv
    permutation_results.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.utils import shuffle

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42   # editable text in PDF
matplotlib.rcParams['ps.fonttype']  = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
MAPPING_DIR   = os.path.join(SCRIPT_DIR, '..', '7.Mapping')
WRAPPER_DIR   = os.path.join(SCRIPT_DIR, '..', '4.Wrapper', 'Fixed_Stacking_Ensemble')
PREPROC_DIR   = os.path.join(SCRIPT_DIR, '..', '1.Loading and Preprocessing')
OUTPUT_DIR    = SCRIPT_DIR

# Load preprocessing helpers from sibling module
sys.path.insert(0, PREPROC_DIR)
import importlib.util as _ilu
_spec   = _ilu.spec_from_file_location(
    "loading_preprocessing",
    os.path.join(PREPROC_DIR, "Loading and preprocessing.py"))
_mod    = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
read_csv_with_encoding = _mod.read_csv_with_encoding
map_categorical_values = _mod.map_categorical_values

# ---------------------------------------------------------------------------
# Global style (consistent with existing DigiLignin figures)
# ---------------------------------------------------------------------------
LABEL_SIZE  = 28
TICK_SIZE   = 24
LEGEND_SIZE = 22
TITLE_SIZE  = 24

PALETTE = {
    'train': '#2166AC',   # blue
    'val':   '#D6604D',   # red-orange
    'test':  '#4DAC26',   # green
}

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper: stratified split (mirrors retrain_best_model.py exactly)
# ---------------------------------------------------------------------------
def stratified_split(X, y, val_size=16, test_size=16, random_state=42):
    np.random.seed(random_state)
    data_with_target = X.copy()
    data_with_target['target'] = y.values.ravel()
    sorted_indices = data_with_target.sort_values('target').index.values
    n_samples = len(sorted_indices)

    val_step = n_samples / val_size
    val_indices = [sorted_indices[int(i * val_step)] for i in range(val_size)]

    remaining = [idx for idx in sorted_indices if idx not in val_indices]
    test_step = len(remaining) / test_size
    test_indices = [remaining[int(i * test_step)] for i in range(test_size)]

    train_indices = [idx for idx in sorted_indices
                     if idx not in val_indices and idx not in test_indices]
    return train_indices, val_indices, test_indices


# ---------------------------------------------------------------------------
# Helper: load dataset and build splits
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_excel(os.path.join(WRAPPER_DIR, 'dataset.xlsx'))
    df_clean = df.dropna(subset=['Tg(deg C)'])

    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    if 'Isocyonate type' in df_clean.columns:
        df_clean = map_categorical_values(df_clean, 'Isocyonate type', isocyanate_mapping)
        df_clean = df_clean.fillna(0)

    FEATURES = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r',
                'Copolyol (wt%)', 'Isocyanate (wt%)']

    # Keep only rows with the required features present
    df_clean = df_clean.dropna(subset=FEATURES + ['Tg(deg C)'])

    X_full = df_clean[['Sample name'] + FEATURES +
                      ['Isocyanate (mmol NCO)', 'Isocyonate type',
                       'tin(II) octoate', 'Sratio(%)']]
    y_full = df_clean[['Tg(deg C)']]

    train_idx, val_idx, test_idx = stratified_split(X_full, y_full)

    X_feats = df_clean[FEATURES]
    y_vec   = df_clean[['Tg(deg C)']]

    X_train = X_feats.loc[train_idx]
    X_val   = X_feats.loc[val_idx]
    X_test  = X_feats.loc[test_idx]
    y_train = y_vec.loc[train_idx]
    y_val   = y_vec.loc[val_idx]
    y_test  = y_vec.loc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test, FEATURES


# ---------------------------------------------------------------------------
# Helper: load saved model artifacts
# ---------------------------------------------------------------------------
def load_model_artifacts():
    base_models = joblib.load(os.path.join(MAPPING_DIR, 'best_model_base_models.joblib'))
    meta_model  = joblib.load(os.path.join(MAPPING_DIR, 'best_model_meta_model.joblib'))
    x_scaler    = joblib.load(os.path.join(MAPPING_DIR, 'best_model_x_scaler.joblib'))
    y_scaler    = joblib.load(os.path.join(MAPPING_DIR, 'best_model_y_scaler.joblib'))
    with open(os.path.join(MAPPING_DIR, 'best_model_metadata.json')) as fh:
        metadata = json.load(fh)
    return base_models, meta_model, x_scaler, y_scaler, metadata


# ---------------------------------------------------------------------------
# Helper: generate ensemble predictions (original scale °C)
# ---------------------------------------------------------------------------
def predict_ensemble(X_scaled, base_models, meta_model, y_scaler):
    meta_feats = np.column_stack([m.predict(X_scaled) for m in base_models])
    y_pred_sc  = meta_model.predict(meta_feats)
    return y_scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()


# ---------------------------------------------------------------------------
# Helper: compute leverage from HAT matrix
# ---------------------------------------------------------------------------
def compute_leverage(X_train_scaled, X_query_scaled):
    """
    h_i = x_i^T (X^T X)^{-1} x_i

    Adds a bias column (intercept) before computing, consistent with
    standard QSPR applicability domain practice.
    """
    ones_train = np.ones((X_train_scaled.shape[0], 1))
    ones_query = np.ones((X_query_scaled.shape[0], 1))

    Xt = np.hstack([ones_train, X_train_scaled])
    Xq = np.hstack([ones_query, X_query_scaled])

    XtX     = Xt.T @ Xt
    # Regularise slightly for numerical stability (ridge-like)
    reg     = 1e-10 * np.eye(XtX.shape[0])
    XtX_inv = np.linalg.inv(XtX + reg)

    leverage = np.einsum('ij,jk,ik->i', Xq, XtX_inv, Xq)
    return leverage


# ---------------------------------------------------------------------------
# PLOT 1 — Williams Plot
# ---------------------------------------------------------------------------
def plot_williams(h_train, z_train,
                  h_val,   z_val,
                  h_test,  z_test,
                  h_star, output_dir):

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Scatter points
    ax.scatter(h_train, z_train,
               color=PALETTE['train'], s=80, alpha=0.75,
               edgecolors='white', linewidths=0.5,
               zorder=3, label=f'Training (n={len(h_train)})')
    ax.scatter(h_val, z_val,
               color=PALETTE['val'], s=120, alpha=0.90,
               edgecolors='white', linewidths=0.5, marker='D',
               zorder=4, label=f'Validation (n={len(h_val)})')
    ax.scatter(h_test, z_test,
               color=PALETTE['test'], s=120, alpha=0.90,
               edgecolors='white', linewidths=0.5, marker='^',
               zorder=4, label=f'Test (n={len(h_test)})')

    # AD boundary lines
    ax.axhline( 3,  color='black', linestyle='--', linewidth=1.5, zorder=2)
    ax.axhline(-3,  color='black', linestyle='--', linewidth=1.5, zorder=2)
    ax.axvline(h_star, color='black', linestyle='--', linewidth=1.5, zorder=2)

    # Shaded AD region
    y_lim_lo, y_lim_hi = ax.get_ylim()
    ax.fill_betweenx([-3, 3], 0, h_star,
                     color='#b0c4de', alpha=0.15, zorder=1,
                     label='Applicability domain')

    # Annotations
    ax.text(h_star * 1.01, 3.2,
            r'$h^* = {:.3f}$'.format(h_star),
            fontsize=LEGEND_SIZE - 2, color='black', va='bottom')
    ax.text(h_star * 1.01, -3.4,
            r'$|z_i| = 3$',
            fontsize=LEGEND_SIZE - 2, color='black', va='top')

    ax.set_xlabel('Leverage  $h_i$', fontsize=LABEL_SIZE)
    ax.set_ylabel('Standardised residual  $z_i$', fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(True, color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9,
              loc='upper right', edgecolor='black')

    plt.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, f'williams_plot.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] williams_plot saved")


# ---------------------------------------------------------------------------
# PLOT 2 — AD Coverage Bar Chart
# ---------------------------------------------------------------------------
def plot_ad_coverage(ad_summary, output_dir):
    subsets = ['Validation', 'Test']
    pct_in  = [ad_summary[s]['pct_inside'] for s in subsets]
    pct_out = [ad_summary[s]['pct_outside'] for s in subsets]

    x   = np.arange(len(subsets))
    w   = 0.45

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    bars_in  = ax.bar(x - w/2, pct_in,  w,
                      color='#2166AC', alpha=0.85, edgecolor='black',
                      label='Inside AD')
    bars_out = ax.bar(x + w/2, pct_out, w,
                      color='#D6604D', alpha=0.85, edgecolor='black',
                      label='Outside AD')

    for bar, val in zip(list(bars_in) + list(bars_out),
                        pct_in + pct_out):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', va='bottom',
                fontsize=TICK_SIZE, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontsize=LABEL_SIZE)
    ax.set_ylabel('Samples (%)', fontsize=LABEL_SIZE)
    ax.set_ylim(0, 115)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.legend(fontsize=LEGEND_SIZE, edgecolor='black')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, f'ad_coverage.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] ad_coverage saved")


# ---------------------------------------------------------------------------
# PLOT 3 — Feature Range Coverage Heatmap
# ---------------------------------------------------------------------------
def plot_feature_coverage(X_train, X_val, X_test, features, output_dir):
    """
    For each feature, shows where val and test samples fall relative to
    the training set range [min, max].  Values > 1 signal extrapolation.
    """
    tr_min = X_train.min()
    tr_max = X_train.max()
    tr_rng = tr_max - tr_min

    def normalise(X):
        return ((X - tr_min) / tr_rng.replace(0, 1)).clip(lower=0, upper=1)

    norm_val  = normalise(X_val)
    norm_test = normalise(X_test)

    # Build summary matrix: rows = samples, cols = features
    all_norm   = pd.concat([norm_val, norm_test], ignore_index=True)
    labels     = (['Val'] * len(norm_val) + ['Test'] * len(norm_test))
    all_norm['Set'] = labels

    val_mean  = norm_val.mean()
    test_mean = norm_test.mean()
    train_mean = normalise(X_train).mean()

    summary_df = pd.DataFrame({
        'Feature': features,
        'Training mean': train_mean.values,
        'Validation mean': val_mean.values,
        'Test mean': test_mean.values,
    })

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    x = np.arange(len(features))
    w = 0.26

    ax.bar(x - w,     summary_df['Training mean'],   w, color='#2166AC',
           alpha=0.85, edgecolor='black', label='Training')
    ax.bar(x,         summary_df['Validation mean'], w, color='#D6604D',
           alpha=0.85, edgecolor='black', label='Validation')
    ax.bar(x + w,     summary_df['Test mean'],       w, color='#4DAC26',
           alpha=0.85, edgecolor='black', label='Test')

    ax.axhline(0.0, color='black', linewidth=1.0, linestyle='-')
    ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--',
               label='Training boundary')

    ax.set_xticks(x)
    short_labels = ['Lignin\n(wt%)', 'Co-polyol\ntype', 'Ratio\nr',
                    'Co-polyol\n(wt%)', 'Isocyanate\n(wt%)']
    ax.set_xticklabels(short_labels, fontsize=TICK_SIZE - 2)
    ax.set_ylabel('Normalised feature value\n(0 = tr. min, 1 = tr. max)',
                  fontsize=LABEL_SIZE - 2)
    ax.set_ylim(-0.05, 1.25)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, edgecolor='black', loc='upper right')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, f'feature_coverage.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  [OK] feature_coverage saved")


# ---------------------------------------------------------------------------
# PLOT 4 — Permutation Test (validation MAE)
# ---------------------------------------------------------------------------
def permutation_test(X_train_sc, y_train, X_val_sc, y_val,
                     base_models, meta_model, y_scaler,
                     n_permutations=1000, output_dir='.'):

    # True validation MAE
    y_val_pred = predict_ensemble(X_val_sc, base_models, meta_model, y_scaler)
    y_val_true = y_val.values.ravel()   # already in °C
    true_mae   = mean_absolute_error(y_val_true, y_val_pred)

    # Permutation distribution
    perm_maes = []
    y_train_orig = y_train.values.ravel()   # already in °C
    for _ in range(n_permutations):
        y_shuffled = shuffle(y_train_orig, random_state=None)
        y_shuffled_sc = y_scaler.transform(y_shuffled.reshape(-1, 1))
        # Re-fit only meta-model on shuffled training meta-features
        meta_feats_train = np.column_stack(
            [m.predict(X_train_sc) for m in base_models])
        from sklearn.linear_model import Ridge as _Ridge
        perm_meta = _Ridge()
        perm_meta.fit(meta_feats_train, y_shuffled_sc.ravel())
        meta_feats_val = np.column_stack(
            [m.predict(X_val_sc) for m in base_models])
        y_perm_pred_sc = perm_meta.predict(meta_feats_val)
        y_perm_pred = y_scaler.inverse_transform(
            y_perm_pred_sc.reshape(-1, 1)).ravel()
        perm_maes.append(mean_absolute_error(y_val_true, y_perm_pred))

    perm_maes = np.array(perm_maes)
    # p-value (standard): fraction of permuted MAEs <= true MAE
    # Interpretation: probability that a null (random) model achieves MAE
    # this low or lower by chance.  p << 0.05 => model is better than chance.
    p_value   = np.mean(perm_maes <= true_mae)

    # Save results
    perm_df = pd.DataFrame({'permuted_MAE': perm_maes})
    perm_df.to_csv(os.path.join(output_dir, 'permutation_results.csv'),
                   index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.hist(perm_maes, bins=40, color='#AABDD4', edgecolor='white',
            alpha=0.85, label=f'Permuted MAE (n={n_permutations})', zorder=2)
    ax.axvline(true_mae, color='#D6604D', linewidth=2.5,
               linestyle='-', zorder=3,
               label=f'True MAE = {true_mae:.2f} °C')
    ax.axvline(np.percentile(perm_maes, 5), color='black',
               linewidth=1.5, linestyle='--', zorder=3,
               label=f'5th percentile = {np.percentile(perm_maes, 5):.2f} °C')

    ax.set_xlabel('Validation MAE (°C)', fontsize=LABEL_SIZE)
    ax.set_ylabel('Frequency', fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # p-value annotation
    ax.text(0.97, 0.95,
            f'p-value = {p_value:.3f}',
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=LEGEND_SIZE,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', alpha=0.85))

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, edgecolor='black')

    plt.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(output_dir, f'permutation_test.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] permutation_test saved  (p = {p_value:.3f})")

    return true_mae, perm_maes, p_value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("APPLICABILITY DOMAIN ANALYSIS — Williams Plot")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading dataset and splitting...")
    X_train, X_val, X_test, y_train, y_val, y_test, FEATURES = load_data()
    print(f"    Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # 2. Load model artifacts
    print("\n[2] Loading saved model artifacts...")
    base_models, meta_model, x_scaler, y_scaler, metadata = load_model_artifacts()
    print(f"    Loaded {len(base_models)} base models")
    print(f"    Metadata val_mae = {metadata['val_mae']:.3f} °C")

    # 3. Scale feature matrices
    X_train_sc = x_scaler.transform(X_train)
    X_val_sc   = x_scaler.transform(X_val)
    X_test_sc  = x_scaler.transform(X_test)

    # 4. Generate predictions (original scale)
    print("\n[3] Generating predictions...")
    y_train_pred = predict_ensemble(X_train_sc, base_models, meta_model, y_scaler)
    y_val_pred   = predict_ensemble(X_val_sc,   base_models, meta_model, y_scaler)
    y_test_pred  = predict_ensemble(X_test_sc,  base_models, meta_model, y_scaler)

    # Raw targets are already in °C from the Excel file — no inverse transform needed
    y_train_true = y_train.values.ravel()
    y_val_true   = y_val.values.ravel()
    y_test_true  = y_test.values.ravel()

    res_train = y_train_true - y_train_pred
    res_val   = y_val_true   - y_val_pred
    res_test  = y_test_true  - y_test_pred

    sigma = np.std(res_train, ddof=1)
    z_train = res_train / sigma
    z_val   = res_val   / sigma
    z_test  = res_test  / sigma

    print(f"    Train MAE = {mean_absolute_error(y_train_true, y_train_pred):.3f} °C")
    print(f"    Val   MAE = {mean_absolute_error(y_val_true,   y_val_pred):.3f} °C")
    print(f"    Test  MAE = {mean_absolute_error(y_test_true,  y_test_pred):.3f} °C")

    # 5. Compute leverage
    print("\n[4] Computing leverage (HAT matrix)...")
    k      = X_train_sc.shape[1]          # 5 features
    n      = X_train_sc.shape[0]          # 104 training samples
    h_star = 3 * (k + 1) / n
    print(f"    k = {k}, n = {n}")
    print(f"    Leverage threshold h* = 3({k}+1)/{n} = {h_star:.4f}")

    h_train = compute_leverage(X_train_sc, X_train_sc)
    h_val   = compute_leverage(X_train_sc, X_val_sc)
    h_test  = compute_leverage(X_train_sc, X_test_sc)

    # 6. AD membership
    def inside_ad(h, z):
        return (h <= h_star) & (np.abs(z) <= 3)

    in_train = inside_ad(h_train, z_train)
    in_val   = inside_ad(h_val,   z_val)
    in_test  = inside_ad(h_test,  z_test)

    ad_summary = {
        'Training':   {'n': len(in_train), 'n_inside': int(in_train.sum()),
                       'pct_inside': 100 * in_train.sum() / len(in_train),
                       'pct_outside': 100 * (~in_train).sum() / len(in_train)},
        'Validation': {'n': len(in_val),   'n_inside': int(in_val.sum()),
                       'pct_inside': 100 * in_val.sum() / len(in_val),
                       'pct_outside': 100 * (~in_val).sum() / len(in_val)},
        'Test':       {'n': len(in_test),  'n_inside': int(in_test.sum()),
                       'pct_inside': 100 * in_test.sum() / len(in_test),
                       'pct_outside': 100 * (~in_test).sum() / len(in_test)},
    }

    print("\n    AD Membership Summary:")
    print(f"    {'Set':<12} {'n':>4} {'Inside':>8} {'%Inside':>9}")
    print("    " + "-" * 38)
    for s, v in ad_summary.items():
        print(f"    {s:<12} {v['n']:>4} {v['n_inside']:>8} {v['pct_inside']:>8.1f}%")

    # 7. Save AD summary CSV
    rows = []
    for s, v in ad_summary.items():
        rows.append({'Set': s, **v})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, 'ad_summary.csv'), index=False)
    print("\n    [OK] ad_summary.csv saved")

    # 8. Generate plots
    print("\n[5] Generating plots...")

    plot_williams(h_train, z_train, h_val, z_val, h_test, z_test,
                  h_star, OUTPUT_DIR)

    plot_ad_coverage(ad_summary, OUTPUT_DIR)

    plot_feature_coverage(X_train, X_val, X_test, FEATURES, OUTPUT_DIR)

    # 9. Permutation test
    print("\n[6] Running permutation test (n=1000)...")
    true_mae, perm_maes, p_value = permutation_test(
        X_train_sc, y_train, X_val_sc, y_val,
        base_models, meta_model, y_scaler,
        n_permutations=1000, output_dir=OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Leverage threshold h*             : {h_star:.4f}")
    print(f"  Standardised residual limit       : ±3")
    print(f"  Training samples inside AD        : {ad_summary['Training']['pct_inside']:.1f}%")
    print(f"  Validation samples inside AD      : {ad_summary['Validation']['pct_inside']:.1f}%")
    print(f"  Test samples inside AD            : {ad_summary['Test']['pct_inside']:.1f}%")
    print(f"  Permutation test p-value          : {p_value:.3f}")
    print(f"  True val MAE                      : {true_mae:.2f} °C")
    print(f"  5th pct of permuted MAE           : {np.percentile(perm_maes, 5):.2f} °C")
    print("=" * 70)
    print("\nAll outputs saved to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
