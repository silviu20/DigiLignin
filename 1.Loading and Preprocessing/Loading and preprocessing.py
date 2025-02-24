
# Standard library imports
import re

# Third-party imports
import chardet
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def detect_file_encoding(file_path):
    """
    Detect the encoding of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Detected encoding
    """
    with open(file_path, 'rb') as file:
        rawdata = file.read(10000)
    return chardet.detect(rawdata)['encoding']


def read_csv_with_encoding(file_path):
    """
    Read a CSV file with automatically detected encoding.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: DataFrame containing the CSV data
    """
    encoding = detect_file_encoding(file_path)
    print(f'Detected encoding: {encoding}')
    return pd.read_csv(file_path, encoding=encoding)


def restructure_dataframe(df):
    """
    Restructure the DataFrame by renaming columns and processing specific fields.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Restructured DataFrame
    """
    # Use first row as header
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header

    # Extract values from complex string patterns
    tg_pattern = r'(-?\d+\.\d+)\s*(?:\((\w*)\))?\s*\[(-?\d+\.\d+)\]'
    tan_pattern = r'(-?\d+\.\d+)\s*\[(-?\d+\.\d+)\]'

    df[['Tg (?C)_1', 'Tg (?C)_2', 'Tg (?C)_3']] = (
        df['Tg (?C)'].str.extract(tg_pattern)
    )
    df[['tan? height_1', 'tan? height_2']] = (
        df['tan? height'].str.extract(tan_pattern)
    )

    # Remove unnecessary columns
    columns_to_drop = ['Tg (?C)', 'Tg (?C)_2', 'tan? height']
    df = df.drop(columns_to_drop, axis=1)

    # Define column order
    column_order = [
        'Sample Name', ' (wt %)', 'HDI', 'Trimer', 'Copolyol',
        'Tg observed', 'Tg or Tm(?C)', '?Cp (J/g·°C)', 'Tg (?C)_1',
        'Tg (?C)_3', 've (mmol/cm3)', 'Mc (Kg/mol)',
        'tan? height_1', 'tan? height_2', 'UTS (Mpa)', '?k (%)',
        'E (Mpa)', 'T5% (?C)', 'Tmax (?C)', 'Residu (%)',
        'Sratio (%)', 'Soluble Fraction (%)'
    ]

    # Reorder columns and replace '-' with NaN
    df = df[column_order]
    df.replace("-", np.nan, inplace=True)

    return df


def encode_categorical_features(df, columns_to_encode):
    """
    Encode categorical features using LabelEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_encode (list): List of columns to encode

    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    encoder = LabelEncoder()
    df_encoded = df.copy()

    for column in columns_to_encode:
        if column in df_encoded.columns:
            df_encoded[column] = encoder.fit_transform(df_encoded[column])

    return df_encoded


def map_categorical_values(df, column, mapping_dict):
    """
    Map categorical values using a dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column to map
        mapping_dict (dict): Mapping dictionary

    Returns:
        pd.DataFrame: DataFrame with mapped values
    """
    df_mapped = df.copy()
    if column in df_mapped.columns:
        df_mapped[column] = df_mapped[column].map(mapping_dict)
    return df_mapped


def scale_numeric_features(df, columns_to_scale):
    """
    Scale numeric features using RobustScaler.

    Args:
        df (pd.DataFrame): Input DataFrame
        columns_to_scale (list): List of columns to scale

    Returns:
        pd.DataFrame: DataFrame with scaled features
    """
    df_scaled = df.copy()
    scaler = RobustScaler()
    
    data_to_scale = df_scaled[columns_to_scale]
    scaled_data = scaler.fit_transform(data_to_scale)
    df_scaled[columns_to_scale] = scaled_data

    return df_scaled


def main():
    """Main function to execute the data processing pipeline."""
    # File path configuration
    file_path = ('C:/Users/P70090917/Desktop/Polyuerthane Lignin/digiLignin Data/'
                 'dataset 2/Processed_1.csv')

    # Read and process data
    df = read_csv_with_encoding(file_path)
    df = df.dropna(subset=['Tg (°C)'])

    # Map categorical values
    isocyanate_mapping = {'N3600': 1, 'HDI': 0, 0: np.nan}
    df_encoded = map_categorical_values(df, 'Isocyanate type', isocyanate_mapping)
    df_encoded = df_encoded.fillna(0)

    # Define columns for scaling
    columns_to_scale = [
        'Lignin (wt%)',
        'Co-polyol (wt%)',
        'Co-polyol type (PTHF)',
        'Isocyanate (wt%)',
        'Isocyanate (mmol NCO)',
        'Isocyanate type',
        'Ratio',
        'Tin(II) octoate',
        'Tg (°C)',
        'Swelling ratio (%)'
    ]

    # Scale features
    df_scaled = scale_numeric_features(df_encoded, columns_to_scale)
    return df_scaled


if __name__ == '__main__':
    df_final = main()