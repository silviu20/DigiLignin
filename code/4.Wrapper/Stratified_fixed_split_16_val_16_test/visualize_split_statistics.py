# -*- coding: utf-8 -*-
"""
Visualize split statistics to show distribution quality
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_split_statistics():
    """Create visualization of split statistics."""
    
    print("Loading split statistics...")
    with open('split_statistics.json', 'r') as f:
        stats = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    
    splits = ['Training', 'Validation', 'Test']
    colors = ['#4C72B0', '#DD8452', '#55A868']
    
    # Data for plotting
    means = [stats['train_target_mean'], stats['val_target_mean'], stats['test_target_mean']]
    stds = [stats['train_target_std'], stats['val_target_std'], stats['test_target_std']]
    mins = [stats['train_target_min'], stats['val_target_min'], stats['test_target_min']]
    maxs = [stats['train_target_max'], stats['val_target_max'], stats['test_target_max']]
    sizes = [stats['train_size'], stats['val_size'], stats['test_size']]
    
    # Plot 1: Mean ± Std Dev
    ax1 = axes[0]
    x_pos = np.arange(len(splits))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Tg (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean ± Std Dev', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(splits, fontsize=11)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}°C\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Min-Max Range
    ax2 = axes[1]
    for i, (split, min_val, max_val, color) in enumerate(zip(splits, mins, maxs, colors)):
        ax2.plot([i, i], [min_val, max_val], 'o-', linewidth=3, markersize=8,
                color=color, label=split)
        # Add horizontal lines for min and max
        ax2.hlines(min_val, i-0.1, i+0.1, colors=color, linewidth=2)
        ax2.hlines(max_val, i-0.1, i+0.1, colors=color, linewidth=2)
        # Add text labels
        ax2.text(i+0.15, min_val, f'{min_val:.1f}°C', va='center', fontsize=9)
        ax2.text(i+0.15, max_val, f'{max_val:.1f}°C', va='center', fontsize=9)
    
    ax2.set_ylabel('Tg (°C)', fontsize=12, fontweight='bold')
    ax2.set_title('Min-Max Range', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(splits, fontsize=11)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Plot 3: Sample Size Distribution
    ax3 = axes[2]
    bars = ax3.bar(x_pos, sizes, alpha=0.7, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('Sample Size Distribution', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(splits, fontsize=11)
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels and percentages
    total_samples = sum(sizes)
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        height = bar.get_height()
        percentage = (size / total_samples) * 100
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle('Stratified Split Statistics: Target Variable (Tg) Distribution', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(f'split_statistics_visualization.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✓ Visualization saved to: split_statistics_visualization.png/pdf/svg")
    
    plt.show()
    
    # Create box plot style comparison
    fig2, ax = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('white')
    
    # Create box plot data
    box_data = []
    for split, mean, std, min_val, max_val in zip(splits, means, stds, mins, maxs):
        # Approximate distribution for visualization
        # Using mean, std, min, max to create representative box plot
        box_data.append({
            'label': split,
            'med': mean,
            'q1': mean - 0.675 * std,  # Approximate Q1
            'q3': mean + 0.675 * std,  # Approximate Q3
            'whislo': min_val,
            'whishi': max_val
        })
    
    positions = [1, 2, 3]
    bp = ax.bxp([box_data[i] for i in range(3)], positions=positions,
                widths=0.6, patch_artist=True, showfliers=False)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style the other elements
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.5)
    
    ax.set_ylabel('Tg (°C)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution Comparison Across Data Splits', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(splits, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add sample size annotations
    for i, (pos, size) in enumerate(zip(positions, sizes)):
        ax.text(pos, ax.get_ylim()[0] - 5, f'n={size}', 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(f'split_distribution_comparison.{ext}', 
                   dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✓ Distribution comparison saved to: split_distribution_comparison.png/pdf/svg")
    
    plt.show()

if __name__ == "__main__":
    visualize_split_statistics()
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

