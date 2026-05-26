import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('closest_inputs_best_model.csv')

# Calculate deviation from perfect prediction line (where Predicted_Tg = Target_Tg)
df['Deviation'] = abs(df['Predicted_Tg'] - df['Target_Tg'])
df['Direction'] = np.where(df['Predicted_Tg'] > df['Target_Tg'], 'Overprediction', 'Underprediction')

# Sort by deviation
df_sorted = df.sort_values('Deviation', ascending=False)

print('DATA POINTS DEVIATING FROM IDEAL PREDICTION LINE')
print('='*60)
print(f'Total points: {len(df)}')
print(f'Training range: -8°C to 96°C')
print()

# Show all points with significant deviation (>1.0°C)
significant_deviation = df[df['Deviation'] > 1.0].copy()
print(f'Points with deviation > 1.0°C: {len(significant_deviation)}')
print()

if len(significant_deviation) > 0:
    print('Significant Deviations (sorted by magnitude):')
    print('-'*60)
    for i, row in significant_deviation.iterrows():
        print(f'Point {i}: Target={row["Target_Tg"]:.2f}°C, Predicted={row["Predicted_Tg"]:.2f}°C')
        print(f'         Deviation: {row["Deviation"]:.2f}°C ({row["Direction"]})')
        print()

# Show extrapolated points specifically
extrapolated = df[(df['Target_Tg'] < -8) | (df['Target_Tg'] > 96)].copy()
print(f'Extrapolated points: {len(extrapolated)}')
print()

if len(extrapolated) > 0:
    print('Extrapolated Points (all deviate significantly):')
    print('-'*60)
    for i, row in extrapolated.iterrows():
        print(f'Point {i}: Target={row["Target_Tg"]:.2f}°C, Predicted={row["Predicted_Tg"]:.2f}°C')
        print(f'         Deviation: {row["Deviation"]:.2f}°C ({row["Direction"]})')
        print()

# Summary statistics
print('Summary Statistics:')
print('-'*60)
print(f'Mean deviation: {df["Deviation"].mean():.2f}°C')
print(f'Max deviation: {df["Deviation"].max():.2f}°C')
print(f'Points with <0.5°C deviation: {len(df[df["Deviation"] < 0.5])}')
print(f'Points with 0.5-1.0°C deviation: {len(df[(df["Deviation"] >= 0.5) & (df["Deviation"] <= 1.0)])}')
print(f'Points with >1.0°C deviation: {len(df[df["Deviation"] > 1.0])}')

# Show points in zoom box regions
print()
print('Points in Zoom Box Regions:')
print('-'*60)

# Bottom zoom box (-5 to 15)
bottom_zoom = df[(df['Target_Tg'] >= -5) & (df['Target_Tg'] <= 15) & 
                (df['Predicted_Tg'] >= -5) & (df['Predicted_Tg'] <= 15)]
print(f'Bottom zoom box (-5 to 15): {len(bottom_zoom)} points')
for i, row in bottom_zoom.iterrows():
    if row['Deviation'] > 0.1:  # Only show points with some deviation
        print(f'  ({row["Target_Tg"]:.1f}, {row["Predicted_Tg"]:.1f}) - Dev: {row["Deviation"]:.2f}°C')

# Top zoom box (70 to 80)
top_zoom = df[(df['Target_Tg'] >= 70) & (df['Target_Tg'] <= 80) & 
              (df['Predicted_Tg'] >= 70) & (df['Predicted_Tg'] <= 80)]
print(f'Top zoom box (70 to 80): {len(top_zoom)} points')
for i, row in top_zoom.iterrows():
    if row['Deviation'] > 0.1:  # Only show points with some deviation
        print(f'  ({row["Target_Tg"]:.1f}, {row["Predicted_Tg"]:.1f}) - Dev: {row["Deviation"]:.2f}°C')
