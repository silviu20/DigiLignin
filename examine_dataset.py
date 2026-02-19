import pandas as pd

# Load the dataset
df = pd.read_excel('dataset.csv.xlsx')

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget variable (Tg) statistics:")
if 'Tg (°C)' in df.columns:
    print(df['Tg (°C)'].describe())
elif 'Tg' in df.columns:
    print(df['Tg'].describe())
else:
    print("Tg column not found. Available columns:", list(df.columns))
