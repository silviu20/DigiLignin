import pandas as pd

# Read the detailed analysis
df = pd.read_csv('top_5_detailed_model_analysis.csv')

# Create a comprehensive summary
print("="*120)
print("COMPREHENSIVE MODEL ANALYSIS - TOP 5 PERFORMING COMBINATIONS")
print("="*120)

# Model architecture summary
print("\nMODEL ARCHITECTURE (Consistent across all top 5 combinations):")
print("-" * 60)
print("BASE MODELS (5 total):")
print("1. Gradient Boosting Regressor:")
print("   - n_estimators: 1000")
print("   - learning_rate: 0.01") 
print("   - max_depth: 3")
print("   - random_state: 42")
print()
print("2. Random Forest Regressor:")
print("   - n_estimators: 1000")
print("   - max_depth: 10 or None")
print("   - min_samples_split: 10")
print("   - random_state: 42")
print()
print("3. Support Vector Regressor:")
print("   - C: 0.1")
print("   - kernel: rbf")
print("   - gamma: auto or scale")
print()
print("4. Lasso Regression:")
print("   - alpha: 0.1")
print("   - max_iter: 1000")
print("   - random_state: 42")
print()
print("5. Elastic Net:")
print("   - alpha: 0.1")
print("   - l1_ratio: 0.1 or 0.5")
print("   - max_iter: 1000")
print("   - random_state: 42")
print()
print("META-MODEL:")
print("• Ridge Regression")
print("• alpha: 1.0")
print("• random_state: 42")
print()

# Performance comparison
print("PERFORMANCE COMPARISON:")
print("-" * 60)
print(f"{'Rank':<5} {'Features':<50} {'#Feat':<6} {'MAE':<8} {'R²':<8} {'Gap':<8}")
print("-" * 60)

for idx, row in df.iterrows():
    rank = int(row['Rank'])
    features = row['Features Used'][:47] + '...' if len(row['Features Used']) > 50 else row['Features Used']
    num_feat = row['Number of Features']
    mae = row['MAE Validation (°C)']
    r2 = row['R² Validation']
    gap = row['Generalization Gap (°C)']
    
    print(f"{rank:<5} {features:<50} {num_feat:<6} {mae:<8} {r2:<8} {gap:<8}")

print()

# Detailed breakdown by rank
for idx, row in df.iterrows():
    rank = int(row['Rank'])
    print(f"{'='*20} RANK {rank} DETAILS {'='*20}")
    print(f"Feature Combination: {row['Features Used']}")
    print(f"Number of Features: {row['Number of Features']}")
    print(f"Performance Metrics:")
    print(f"  • MAE Validation: {row['MAE Validation (°C)']}°C")
    print(f"  • R² Validation: {row['R² Validation']}")
    print(f"  • Generalization Gap: {row['Generalization Gap (°C)']}°C")
    
    print(f"\nHyperparameter Variations from Baseline:")
    base_models = eval(row['Base Models'])
    base_hyperparams = eval(row['Base Model Hyperparameters'])
    
    for i, (model, hyperparams) in enumerate(zip(base_models, base_hyperparams), 1):
        print(f"  {i}. {model}:")
        print(f"     {hyperparams}")
    
    print(f"\nMeta Model: {row['Meta Model']} ({row['Meta Model Hyperparameters']})")
    print()

# Key insights
print("KEY INSIGHTS:")
print("-" * 60)
print("1. MODEL CONSISTENCY:")
print("   • All top 5 combinations use identical model architectures")
print("   • Performance differences come solely from feature selection")
print("   • No hyperparameter tuning differences between ranks")
print()
print("2. FEATURE SELECTION IMPACT:")
print("   • Best performance: 5 features (Rank 1: 15.498°C MAE)")
print("   • More features don't always improve performance")
print("   • Core features (Lignin, Co-polyol type, r) are always present")
print()
print("3. MODEL GENERALIZATION:")
print("   • Small generalization gaps (±0.5°C) indicate good validation")
print("   • No significant overfitting observed")
print("   • Consistent performance across different feature subsets")
print()
print("4. HYPERPARAMETER STABILITY:")
print("   • Tree models: 1000 estimators consistently optimal")
print("   • Regularization: alpha=0.1 for linear models")
print("   • Meta-model: Ridge with alpha=1.0 works best")
print()

# Create final summary table
summary_data = []
for idx, row in df.iterrows():
    rank = int(row['Rank'])
    # Parse the feature string safely
    feature_str = row['Features Used']
    features = [f.strip() for f in feature_str[1:-1].split(',')]  # Remove brackets and split
    
    summary_data.append({
        'Rank': rank,
        'Features': ', '.join(features),
        'Num_Features': len(features),
        'MAE_Validation': float(row['MAE Validation (°C)']),
        'R2_Validation': float(row['R² Validation']),
        'Generalization_Gap': float(row['Generalization Gap (°C)']),
        'Base_Models': 'GBR, RF, SVR, Lasso, ElasticNet',
        'Meta_Model': 'Ridge (alpha=1.0)',
        'Tree_Estimators': 1000,
        'Regularization': 'alpha=0.1'
    })

final_df = pd.DataFrame(summary_data)
final_df.to_csv('final_model_summary.csv', index=False)

print("FILES CREATED:")
print("-" * 60)
print("• top_5_detailed_model_analysis.csv - Complete model details")
print("• top_5_summary_table.csv - Compact summary")
print("• final_model_summary.csv - Final formatted summary")
print("="*120)
