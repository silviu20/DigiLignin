# -*- coding: utf-8 -*-
"""
Mapping script for the best model identified from wrapper experiments
Model: 10 base estimators with 5 features
Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)

This script generates predictions across the entire feature space to create a mapping
of how different input combinations affect the glass transition temperature (Tg).
"""

import numpy as np
import pandas as pd
import itertools
import joblib
import json
from datetime import datetime

def map_target_batch(base_models, meta_model, x_scaler, y_scaler, batch_size=10000):
    """
    Map the target variable (Tg) across the feature space using batch processing.
    
    Args:
        base_models: List of trained base models
        meta_model: Trained meta-model (Ridge)
        x_scaler: Fitted scaler for input features
        y_scaler: Fitted scaler for target variable
        batch_size: Number of combinations to process before saving intermediate results
    
    Returns:
        pd.DataFrame: Results with all feature combinations and predicted Tg values
    """
    
    # Define feature value ranges for mapping
    # Based on the 5 features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
    
    feature_values = [
        np.arange(0, 70, 1),              # 'Lignin (wt%)' - 0 to 70% in 1% steps
        [250, 650, 1000],                 # 'Co-polyol type (PTHF)' - discrete values
        np.arange(0.6, 1.4 + 0.05, 0.05), # 'r' - ratio from 0.6 to 1.4 in 0.05 steps
        np.arange(0, 66 + 2, 2),          # 'Copolyol (wt%)' - 0 to 66% in 2% steps
        np.arange(0, 20 + 0.5, 0.5),      # 'Isocyanate (wt%)' - 0 to 20% in 0.5% steps
    ]
    
    # Define input columns (must match the order used in training)
    input_columns = ['Lignin (wt%)', 'Co-polyol type (PTHF)', 'r', 'Copolyol (wt%)', 'Isocyanate (wt%)']
    
    # All columns including the target
    all_columns = input_columns + ['Tg (°C)']
    
    results = pd.DataFrame(columns=all_columns)
    
    # Calculate total combinations
    total_combinations = np.prod([len(fv) for fv in feature_values])
    print(f"Total combinations to process: {total_combinations:,}")
    print(f"Estimated memory usage: ~{total_combinations * len(all_columns) * 8 / 1e6:.1f} MB")
    print(f"Processing in batches of {batch_size:,} combinations")
    print("="*80)
    
    batch = []
    start_time = datetime.now()
    
    for i, combo in enumerate(itertools.product(*feature_values)):
        # Progress reporting
        if i % batch_size == 0 and i > 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed
            remaining = (total_combinations - i) / rate
            print(f"Progress: {i:,}/{total_combinations:,} ({100*i/total_combinations:.1f}%) | "
                  f"Rate: {rate:.0f} combo/s | ETA: {remaining/60:.1f} min")
        
        # Create input data for this combination
        input_data = pd.DataFrame([combo], columns=input_columns)
        
        # Scale the input
        input_data_scaled = x_scaler.transform(input_data)
        
        # Generate base model predictions (meta-features)
        base_predictions = np.column_stack([model.predict(input_data_scaled) for model in base_models])
        
        # Generate final prediction using meta-model
        prediction_scaled = meta_model.predict(base_predictions)
        
        # Inverse transform to get actual Tg value
        prediction = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        # Add to batch
        batch.append((*combo, prediction))
        
        # Save batch when full or at the end
        if (i + 1) % batch_size == 0 or i == total_combinations - 1:
            batch_df = pd.DataFrame(batch, columns=all_columns)
            results = pd.concat([results, batch_df], ignore_index=True)
            batch = []
    
    total_time = (datetime.now() - start_time).total_seconds()
    print("="*80)
    print(f"Mapping completed in {total_time/60:.1f} minutes")
    print(f"Average rate: {total_combinations/total_time:.0f} combinations/second")
    
    return results

def analyze_mapping_results(results):
    """
    Analyze and summarize the mapping results.
    
    Args:
        results: DataFrame with mapping results
    
    Returns:
        dict: Summary statistics
    """
    print("\n" + "="*80)
    print("MAPPING RESULTS SUMMARY")
    print("="*80)
    
    summary = {
        'total_combinations': len(results),
        'tg_min': results['Tg (°C)'].min(),
        'tg_max': results['Tg (°C)'].max(),
        'tg_mean': results['Tg (°C)'].mean(),
        'tg_std': results['Tg (°C)'].std(),
        'tg_median': results['Tg (°C)'].median(),
        'tg_q25': results['Tg (°C)'].quantile(0.25),
        'tg_q75': results['Tg (°C)'].quantile(0.75)
    }
    
    print(f"Total combinations mapped: {summary['total_combinations']:,}")
    print(f"\nPredicted Tg (°C) statistics:")
    print(f"  Min:    {summary['tg_min']:.2f}°C")
    print(f"  Q25:    {summary['tg_q25']:.2f}°C")
    print(f"  Median: {summary['tg_median']:.2f}°C")
    print(f"  Mean:   {summary['tg_mean']:.2f}°C")
    print(f"  Q75:    {summary['tg_q75']:.2f}°C")
    print(f"  Max:    {summary['tg_max']:.2f}°C")
    print(f"  Std:    {summary['tg_std']:.2f}°C")
    print(f"  Range:  {summary['tg_max'] - summary['tg_min']:.2f}°C")
    
    # Feature-wise analysis
    print(f"\nFeature ranges in mapping:")
    for col in results.columns[:-1]:  # Exclude Tg column
        print(f"  {col}:")
        print(f"    Min: {results[col].min():.2f}, Max: {results[col].max():.2f}, "
              f"Unique values: {results[col].nunique()}")
    
    print("="*80)
    
    return summary

def main():
    """Main execution function for mapping."""
    print("="*80)
    print("MAPPING WITH BEST MODEL")
    print("="*80)
    print("Model: 10 base estimators")
    print("Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)")
    print("="*80)
    
    # Load the trained models and scalers
    print("\nLoading trained models and scalers...")
    try:
        base_models = joblib.load('best_model_base_models.joblib')
        print("  ✓ Loaded base_models")
        
        meta_model = joblib.load('best_model_meta_model.joblib')
        print("  ✓ Loaded meta_model")
        
        x_scaler = joblib.load('best_model_x_scaler.joblib')
        print("  ✓ Loaded x_scaler")
        
        y_scaler = joblib.load('best_model_y_scaler.joblib')
        print("  ✓ Loaded y_scaler")
        
        # Load metadata for verification
        with open('best_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print("  ✓ Loaded metadata")
        
        print(f"\nModel information:")
        print(f"  Number of base models: {len(base_models)}")
        print(f"  Features: {metadata['features']}")
        print(f"  Validation R²: {metadata['val_r2']:.4f}")
        print(f"  Test R²: {metadata['test_r2']:.4f}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find model files!")
        print(f"   {e}")
        print("\nPlease run 'retrain_best_model.py' first to train and save the model.")
        return
    
    # Perform mapping
    print("\n" + "="*80)
    print("STARTING MAPPING PROCESS")
    print("="*80)
    
    mapped_results = map_target_batch(base_models, meta_model, x_scaler, y_scaler, batch_size=10000)
    
    # Analyze results
    summary = analyze_mapping_results(mapped_results)
    
    # Save results
    print("\nSaving results...")
    
    # Save main results
    output_filename = 'mapped_results_tg_best_model.csv'
    mapped_results.to_csv(output_filename, index=False)
    print(f"  ✓ Saved mapping results to '{output_filename}'")
    
    # Save summary statistics
    summary_filename = 'mapping_summary.json'
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"  ✓ Saved summary statistics to '{summary_filename}'")
    
    # Save a sample of results for quick inspection
    sample_filename = 'mapped_results_sample.csv'
    sample_size = min(1000, len(mapped_results))
    mapped_results.sample(n=sample_size, random_state=42).to_csv(sample_filename, index=False)
    print(f"  ✓ Saved sample ({sample_size} rows) to '{sample_filename}'")
    
    print("\n" + "="*80)
    print("MAPPING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Output files:")
    print(f"  - {output_filename} (full results)")
    print(f"  - {summary_filename} (summary statistics)")
    print(f"  - {sample_filename} (random sample)")
    print("="*80)

if __name__ == "__main__":
    main()
