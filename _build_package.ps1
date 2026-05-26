# Build script used to assemble the data share package.
# Re-run to rebuild from the parent repository state.
$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSScriptRoot
$share = $PSScriptRoot

# --- 1. Copy code (.py, .md, .txt, .json, .ipynb) preserving structure ---
$codeFolders = @(
    "1.Loading and Preprocessing",
    "2.Correlation",
    "3.Partial dependence plots",
    "3.PCA",
    "4.Wrapper",
    "5.Model",
    "6.Model metrics",
    "7.Mapping",
    "8.Extrapolation",
    "9.Parallel coordinates plot",
    "10.Dataset Distribution based on swelling ratio",
    "Universality",
    "Graphical abstract"
)
$codeExt = @("*.py","*.md","*.txt","*.json","*.ipynb","*.tex")
foreach ($folder in $codeFolders) {
    $src = Join-Path $root $folder
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path (Join-Path $share "code") $folder
    $null = New-Item -ItemType Directory -Force -Path $dst
    Get-ChildItem $src -Recurse -File -Include $codeExt |
        Where-Object { $_.FullName -notmatch '__pycache__|\\venv\\' } |
        ForEach-Object {
            $rel = $_.FullName.Substring($src.Length).TrimStart('\')
            $target = Join-Path $dst $rel
            $null = New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent)
            Copy-Item $_.FullName $target -Force
        }
}

# --- 2. Copy raw dataset ---
$dataDst = Join-Path $share "data"
Copy-Item (Join-Path $root "dataset.csv.xlsx") (Join-Path $dataDst "dataset.csv.xlsx") -Force
$wrapperXlsx = Join-Path $root "4.Wrapper\Fixed_Stacking_Ensemble\dataset.xlsx"
if (Test-Path $wrapperXlsx) {
    Copy-Item $wrapperXlsx (Join-Path $dataDst "dataset.xlsx") -Force
}

# --- 3. Copy best model artifacts ---
$modelDst = Join-Path $share "models\best_model"
$bestArtifacts = @(
    "7.Mapping\best_model_base_models.joblib",
    "7.Mapping\best_model_meta_model.joblib",
    "7.Mapping\best_model_x_scaler.joblib",
    "7.Mapping\best_model_y_scaler.joblib",
    "7.Mapping\best_model_features.txt",
    "7.Mapping\best_model_metadata.json"
)
foreach ($a in $bestArtifacts) {
    $src = Join-Path $root $a
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $modelDst (Split-Path $a -Leaf)) -Force
    }
}

# --- 4. Copy key result tables (small/medium CSVs only) ---
$resultsDst = Join-Path $share "results"
$resultFiles = @(
    "4.Wrapper\final_model_summary.csv",
    "4.Wrapper\top_5_combinations_table.csv",
    "4.Wrapper\top_5_summary_table.csv",
    "4.Wrapper\top_5_detailed_model_analysis.csv",
    "4.Wrapper\top_5_combinations_latex.txt",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\model6_n_estimators_results.csv",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\model6_plot_data.csv",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\top_5_best_models.csv",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\top_5_best_models_no_sratio.csv",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\final_validation_summary.csv",
    "4.Wrapper\Fixed_stacking_ensemble_with_n_estimators\all_combinations_n_estimators_results.csv",
    "4.Wrapper\Stratified_fixed_split_16_val_16_test\fixed_split_results.csv",
    "4.Wrapper\Stratified_fixed_split_16_val_16_test\top_10_models_validation_with_test.csv",
    "4.Wrapper\Stratified_fixed_split_16_val_16_test\top_10_models_table.csv",
    "4.Wrapper\Stratified_fixed_split_16_val_16_test\split_statistics.json",
    "4.Wrapper\Stratified_fixed_split_16_val_16_test\split_statistics_table.csv",
    "6.Model metrics\Fixed Stacked Ensemble\Model6_700_Individual_Models_Performance.csv",
    "6.Model metrics\Fixed Stacked Ensemble\Model6_700_Regression_Data.csv",
    "6.Model metrics\Fixed Stacked Ensemble\Best_Performing_Model_Analysis\Best_Model_Detailed_Report.csv",
    "6.Model metrics\Stratified Stacked Ensemble\individual_models_performance.csv",
    "7.Mapping\mapping_summary.json",
    "7.Mapping\mapping_summary_fast.json",
    "7.Mapping\mapped_results_sample.csv",
    "7.Mapping\mapped_results_sample_fast.csv",
    "8.Extrapolation\closest_inputs_best_model.csv",
    "Universality\ad_summary.csv",
    "Universality\permutation_results.csv"
)
foreach ($r in $resultFiles) {
    $src = Join-Path $root $r
    if (Test-Path $src) {
        $target = Join-Path $resultsDst $r
        $null = New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent)
        Copy-Item $src $target -Force
    }
}

# --- 5. Copy headline figures (one format each) ---
$figDst = Join-Path $share "figures"
$figFiles = @(
    "6.Model metrics\Ensemble_Predicted_vs_Actual.png",
    "6.Model metrics\Fixed Stacked Ensemble\Model6_700_Regression_Wrapper.png",
    "6.Model metrics\Fixed Stacked Ensemble\Model6_n_estimators_performance.png",
    "6.Model metrics\Fixed Stacked Ensemble\Best_Performing_Model_Analysis\Best_Model_Comprehensive_Analysis.png",
    "7.Mapping\density_plots_best_model\merged_density_plots.png",
    "7.Mapping\distribution_tg_best_model.png",
    "8.Extrapolation\Target_Predicted_Regression_Plot_Replica.png",
    "Universality\williams_plot.png",
    "Universality\ad_coverage.png",
    "Universality\permutation_test.png",
    "Universality\feature_coverage.png",
    "4.Wrapper\stacking_ensemble_summary.png"
)
foreach ($f in $figFiles) {
    $src = Join-Path $root $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $figDst (Split-Path $f -Leaf)) -Force
    }
}

Write-Host "Build complete: $share"
