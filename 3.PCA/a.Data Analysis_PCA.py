# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:30:13 2024

@author: P70090917
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming df_scaled is your DataFrame with already scaled data
# Select only numeric columns for PCA
numeric_columns = df_scaled.select_dtypes(include=['float64', 'int64']).columns
X = df_scaled[numeric_columns]

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Print the explained variance ratio for each component
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1} explained variance ratio: {ratio:.4f}")

# Select the number of components based on the desired explained variance
# For example, to explain 95% of the variance:
n_components = next(i for i, ratio in enumerate(cumulative_variance_ratio) if ratio >= 0.95) + 1

print(f"\nNumber of components needed to explain 95% of variance: {n_components}")

# Apply PCA with the selected number of components
pca_final = PCA(n_components=n_components)
X_pca_final = pca_final.fit_transform(X)

# Create a DataFrame with the selected principal components
pca_df = pd.DataFrame(data=X_pca_final, columns=[f'PC{i+1}' for i in range(n_components)])

# Add the non-numeric columns back to the PCA DataFrame
non_numeric_columns = df_scaled.select_dtypes(exclude=['float64', 'int64']).columns
final_df = pd.concat([df_scaled[non_numeric_columns].reset_index(drop=True), pca_df], axis=1)

print("\nFinal DataFrame shape:", final_df.shape)
print("\nFinal DataFrame columns:")
print(final_df.columns)

# Print the feature importance (loadings) for each principal component
feature_importance = pd.DataFrame(
    pca_final.components_.T,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=numeric_columns
)
print("\nFeature importance (loadings) for each principal component:")
print(feature_importance)


# import pandas as pd
# import numpy as np

# # Assuming you have the loadings in a DataFrame called 'feature_importance'
# # If not, recreate it from the data you provided
# feature_importance = pd.DataFrame({
#     'PC1': [-0.212650, 0.366080, 0.127948, -0.331452, -0.479002, -0.005541, -0.285840, -0.162429, -0.187297, 0.567355],
#     'PC2': [0.396131, -0.245592, -0.072793, -0.171899, 0.275564, -0.269863, -0.232393, 0.493993, 0.151680, 0.527363],
#     'PC3': [-0.368462, 0.273668, -0.097754, 0.069636, 0.531573, -0.195904, 0.521149, -0.060652, -0.191406, 0.376931],
#     'PC4': [0.174537, -0.376409, 0.094284, 0.419905, -0.155091, 0.351871, 0.250603, -0.369091, 0.219073, 0.497749],
#     'PC5': [-0.333197, 0.004785, -0.454574, 0.490149, -0.017641, 0.342012, -0.355917, 0.367494, -0.230757, 0.099055]
# }, index=['Lignin (wt%)', 'Co-polyol (wt%)', 'Co-polyol type (PTHF)', 'Isocyanate (wt%)', 
#           'Isocyanate (mmol NCO)', 'Isocyanate type', 'Ratio', 'Tin(II) octoate', 'Tg (°C)', 'Swelling ratio (%)'])

# Calculate the average absolute loading across all PCs for each feature
avg_importance = np.abs(feature_importance).mean(axis=1).sort_values(ascending=False)

print("Features ranked by average absolute loading across all PCs:")
print(avg_importance)

# Calculate the maximum absolute loading for each feature
max_importance = np.abs(feature_importance).max(axis=1).sort_values(ascending=False)

print("\nFeatures ranked by maximum absolute loading across all PCs:")
print(max_importance)

# Identify top features for each PC
top_features_per_pc = pd.DataFrame({
    f'PC{i+1}': feature_importance[f'PC{i+1}'].abs().nlargest(5).index.tolist()
    for i in range(5)
})

print("\nTop 5 most important features for each PC:")
print(top_features_per_pc)

# # Calculate cumulative explained variance
# explained_variance_ratio = [0.3355, 0.2377, 0.1633, 0.1406, 0.0883]
# cumulative_variance = np.cumsum(explained_variance_ratio)

# print("\nCumulative explained variance:")
# for i, var in enumerate(cumulative_variance):
#     print(f"PC1 to PC{i+1}: {var:.4f}")
