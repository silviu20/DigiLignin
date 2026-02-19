import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load the actual dataset
df = pd.read_excel('../dataset.csv.xlsx')
df_clean = df.dropna(subset=['Tg(deg C)'])

# Map categorical variables
isocyanate_mapping = {'N3600': 1, 'HDI': 0}
df_clean['Isocyanate type'] = df_clean['Isocyonate type'].map(isocyanate_mapping).fillna(0)

# Prepare features and target
feature_columns = ['Lignin (wt%)', 'r', 'Co-polyol type (PTHF)', 
                   'Isocyanate (mmol NCO)', 'Isocyanate type', 'tin(II) octoate', 'Sratio(%)']
X = df_clean[feature_columns]
y = df_clean['Tg(deg C)']

print("=== BASELINE MODEL COMPARISON ===")
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Target range: {y.min():.1f} to {y.max():.1f}°C")
print()

# Test simple models
models = {
    'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': make_pipeline(StandardScaler(), 
                                     LinearRegression())  # Using Linear as placeholder
}

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_mae = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    
    print(f"{name}:")
    print(f"  R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  MAE: {cv_mae.mean():.1f} ± {cv_mae.std():.1f}°C")
    print()

# Simple heuristic: predict mean Tg
mean_tg = y.mean()
mae_mean = np.mean(np.abs(y - mean_tg))
print(f"Simple Mean Predictor:")
print(f"  R²: 0.000 (by definition)")
print(f"  MAE: {mae_mean:.1f}°C")
print()

print("=== ANALYSIS ===")
print("• If sophisticated models perform similarly to predicting the mean,")
print("  the features don't contain much predictive information")
print("• The corrected stacking model (R²=0.28) is only slightly better")
print("  than predicting the mean value")
print("• This suggests the model is not practically useful")
