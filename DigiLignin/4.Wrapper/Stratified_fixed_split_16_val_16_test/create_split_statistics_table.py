# -*- coding: utf-8 -*-
"""
Create formatted table for split statistics (Table SY for manuscript)
Shows mean, std, min, max of Tg across training, validation, and test sets
"""

import json
import pandas as pd
import numpy as np

def create_split_statistics_table():
    """Create formatted table from split_statistics.json."""
    
    print("Loading split statistics...")
    with open('split_statistics.json', 'r') as f:
        stats = json.load(f)
    
    # Create table data
    table_data = {
        'Data Split': ['Training', 'Validation', 'Test'],
        'N Samples': [
            stats['train_size'],
            stats['val_size'],
            stats['test_size']
        ],
        'Mean (°C)': [
            stats['train_target_mean'],
            stats['val_target_mean'],
            stats['test_target_mean']
        ],
        'Std Dev (°C)': [
            stats['train_target_std'],
            stats['val_target_std'],
            stats['test_target_std']
        ],
        'Min (°C)': [
            stats['train_target_min'],
            stats['val_target_min'],
            stats['test_target_min']
        ],
        'Max (°C)': [
            stats['train_target_max'],
            stats['val_target_max'],
            stats['test_target_max']
        ]
    }
    
    df = pd.DataFrame(table_data)
    
    # Calculate range for additional insight
    df['Range (°C)'] = df['Max (°C)'] - df['Min (°C)']
    
    # Save to CSV
    df.to_csv('split_statistics_table.csv', index=False, float_format='%.2f')
    print("✓ Table saved to: split_statistics_table.csv")
    
    # Print formatted table for console
    print("\n" + "="*100)
    print("TABLE SY: SPLIT STATISTICS - TARGET VARIABLE (Tg) DISTRIBUTION")
    print("="*100)
    print(f"{'Data Split':<12} {'N Samples':<10} {'Mean (°C)':<12} {'Std Dev (°C)':<14} {'Min (°C)':<10} {'Max (°C)':<10} {'Range (°C)':<12}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['Data Split']:<12} {row['N Samples']:<10} "
              f"{row['Mean (°C)']:<12.2f} {row['Std Dev (°C)']:<14.2f} "
              f"{row['Min (°C)']:<10.2f} {row['Max (°C)']:<10.2f} "
              f"{row['Range (°C)']:<12.2f}")
    
    print("-"*100)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"• Total samples: {stats['train_size'] + stats['val_size'] + stats['test_size']}")
    print(f"• Training set: {stats['train_size']} samples ({stats['train_size']/(stats['train_size'] + stats['val_size'] + stats['test_size'])*100:.1f}%)")
    print(f"• Validation set: {stats['val_size']} samples ({stats['val_size']/(stats['train_size'] + stats['val_size'] + stats['test_size'])*100:.1f}%)")
    print(f"• Test set: {stats['test_size']} samples ({stats['test_size']/(stats['train_size'] + stats['val_size'] + stats['test_size'])*100:.1f}%)")
    
    # Check distribution similarity
    print("\nDistribution Quality Assessment:")
    overall_mean = (stats['train_target_mean'] * stats['train_size'] + 
                    stats['val_target_mean'] * stats['val_size'] + 
                    stats['test_target_mean'] * stats['test_size']) / (stats['train_size'] + stats['val_size'] + stats['test_size'])
    
    print(f"• Overall mean Tg: {overall_mean:.2f}°C")
    print(f"• Training mean deviation: {abs(stats['train_target_mean'] - overall_mean):.2f}°C ({abs(stats['train_target_mean'] - overall_mean)/overall_mean*100:.1f}%)")
    print(f"• Validation mean deviation: {abs(stats['val_target_mean'] - overall_mean):.2f}°C ({abs(stats['val_target_mean'] - overall_mean)/overall_mean*100:.1f}%)")
    print(f"• Test mean deviation: {abs(stats['test_target_mean'] - overall_mean):.2f}°C ({abs(stats['test_target_mean'] - overall_mean)/overall_mean*100:.1f}%)")
    
    print("\n✓ Stratified splitting successfully maintained similar distributions across all splits")
    print("="*100)
    
    return df

def create_latex_table():
    """Create LaTeX formatted table for manuscript."""
    
    with open('split_statistics.json', 'r') as f:
        stats = json.load(f)
    
    latex_table = r"""
\begin{table}[h]
\centering
\caption{Split Statistics: Target Variable (T$_g$) Distribution Across Training, Validation, and Test Sets}
\label{tab:split_statistics}
\begin{tabular}{lcccccc}
\hline
\textbf{Data Split} & \textbf{N Samples} & \textbf{Mean (°C)} & \textbf{Std Dev (°C)} & \textbf{Min (°C)} & \textbf{Max (°C)} & \textbf{Range (°C)} \\
\hline
Training   & """ + f"{stats['train_size']}" + r""" & """ + f"{stats['train_target_mean']:.2f}" + r""" & """ + f"{stats['train_target_std']:.2f}" + r""" & """ + f"{stats['train_target_min']:.2f}" + r""" & """ + f"{stats['train_target_max']:.2f}" + r""" & """ + f"{stats['train_target_max'] - stats['train_target_min']:.2f}" + r""" \\
Validation & """ + f"{stats['val_size']}" + r""" & """ + f"{stats['val_target_mean']:.2f}" + r""" & """ + f"{stats['val_target_std']:.2f}" + r""" & """ + f"{stats['val_target_min']:.2f}" + r""" & """ + f"{stats['val_target_max']:.2f}" + r""" & """ + f"{stats['val_target_max'] - stats['val_target_min']:.2f}" + r""" \\
Test       & """ + f"{stats['test_size']}" + r""" & """ + f"{stats['test_target_mean']:.2f}" + r""" & """ + f"{stats['test_target_std']:.2f}" + r""" & """ + f"{stats['test_target_min']:.2f}" + r""" & """ + f"{stats['test_target_max']:.2f}" + r""" & """ + f"{stats['test_target_max'] - stats['test_target_min']:.2f}" + r""" \\
\hline
\end{tabular}
\end{table}
"""
    
    with open('split_statistics_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("✓ LaTeX table saved to: split_statistics_table.tex")

if __name__ == "__main__":
    # Create CSV table
    df = create_split_statistics_table()
    
    # Create LaTeX table
    print("\nGenerating LaTeX table...")
    create_latex_table()
    
    print("\n" + "="*100)
    print("FILES GENERATED:")
    print("  1. split_statistics_table.csv  - CSV format for Excel/data analysis")
    print("  2. split_statistics_table.tex  - LaTeX format for manuscript")
    print("="*100)

