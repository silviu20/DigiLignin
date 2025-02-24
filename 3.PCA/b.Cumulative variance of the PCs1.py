#!/usr/bin/env python3
"""
PCA Analysis and Visualization Script.

Creates visualizations of PCA component analysis with cumulative
and individual variance ratios.

Created on Mon Dec 2 14:23:36 2024
Author: P70090917
"""

# Standard library imports
from typing import Tuple

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def perform_pca(df: pd.DataFrame) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA on numeric columns of the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with scaled values

    Returns:
        Tuple[PCA, np.ndarray]: Fitted PCA object and transformed data
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    pca = PCA()
    x_pca = pca.fit_transform(df[numeric_columns])
    return pca, x_pca


def setup_plot() -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Set up the plot with two y-axes.

    Returns:
        Tuple[plt.Figure, plt.Axes, plt.Axes]: Figure and both axes objects
    """
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()
    return fig, ax1, ax2


def plot_cumulative_variance(
    ax: plt.Axes,
    cumulative_variance: np.ndarray,
    n_components: int,
    threshold: float
) -> None:
    """
    Plot cumulative variance line with annotations.

    Args:
        ax (plt.Axes): Main axes object
        cumulative_variance (np.ndarray): Cumulative variance ratios
        n_components (int): Number of components for threshold
        threshold (float): Variance threshold
    """
    ax.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        'bo-',
        linewidth=3,
        markersize=10,
        label='Cumulative Explained Variance'
    )

    # Add threshold lines
    threshold_intersect = cumulative_variance[n_components - 1]
    ax.axhline(
        y=threshold_intersect,
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'More than {threshold*100}% Variance Threshold'
    )
    ax.axvline(x=n_components, color='orange', linestyle='--', linewidth=2)

    # Add annotations
    for i, variance in enumerate(cumulative_variance):
        color = 'orange' if i + 1 == n_components else 'black'
        weight = 'bold' if i + 1 == n_components else 'normal'
        ax.annotate(
            f'{variance*100:.1f}%',
            xy=(i + 1, variance),
            xytext=(0, 15),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=16,
            color=color,
            weight=weight
        )


def plot_individual_variance(
    ax: plt.Axes,
    individual_variance: np.ndarray
) -> None:
    """
    Plot individual variance bars with annotations.

    Args:
        ax (plt.Axes): Secondary axes object
        individual_variance (np.ndarray): Individual variance ratios
    """
    bars = ax.bar(
        range(1, len(individual_variance) + 1),
        individual_variance,
        alpha=0.2,
        color='gray',
        label='Individual Variance'
    )

    for i, variance in enumerate(individual_variance):
        ax.text(
            i + 1,
            variance,
            f'{variance*100:.1f}%',
            ha='center',
            va='bottom',
            fontsize=16,
            color='gray'
        )


def customize_axes(
    ax1: plt.Axes,
    ax2: plt.Axes,
    cumulative_variance: np.ndarray,
    individual_variance: np.ndarray
) -> None:
    """
    Customize axes appearance and labels.

    Args:
        ax1 (plt.Axes): Main axes object
        ax2 (plt.Axes): Secondary axes object
        cumulative_variance (np.ndarray): Cumulative variance ratios
        individual_variance (np.ndarray): Individual variance ratios
    """
    # Primary axis customization
    ax1.set_xlabel('Number of Components', fontsize=18)
    ax1.set_ylabel('Cumulative Explained Variance Ratio', fontsize=18)
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xticks(range(1, len(cumulative_variance) + 1))

    # Secondary axis customization
    ax2.set_ylabel('Individual Explained Variance Ratio', fontsize=18)
    ax2.set_ylim(0, max(individual_variance) * 1.2)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='lower right',
        fontsize=18
    )


def print_summary(
    individual_variance: np.ndarray,
    cumulative_variance: np.ndarray,
    n_components: int,
    threshold: float
) -> None:
    """
    Print summary statistics of PCA analysis.

    Args:
        individual_variance (np.ndarray): Individual variance ratios
        cumulative_variance (np.ndarray): Cumulative variance ratios
        n_components (int): Number of components for threshold
        threshold (float): Variance threshold
    """
    print("\nPCA Component Analysis:")
    print("-" * 80)
    print(f"{'Component':^10} {'Individual Variance':^20} {'Cumulative Variance':^20}")
    print("-" * 80)
    
    for i, (ind_var, cum_var) in enumerate(zip(individual_variance, cumulative_variance)):
        print(f"{i+1:^10d} {ind_var*100:^20.2f}% {cum_var*100:^20.2f}%")
    
    print("-" * 80)
    print(f"\nComponents needed for {threshold*100}% variance: {n_components}")


def save_plot(formats: list = None) -> None:
    """
    Save the plot in specified formats.

    Args:
        formats (list, optional): List of formats to save in. Defaults to ['tiff'].
    """
    if formats is None:
        formats = ['tiff']

    for fmt in formats:
        dpi = 600 if fmt in ['png', 'tiff'] else None
        plt.savefig(
            f'PCA_cumulative_variance.{fmt}',
            dpi=dpi,
            bbox_inches='tight'
        )


def plot_pca_cumulative_variance(
    df_scaled: pd.DataFrame,
    variance_threshold: float = 0.90
) -> int:
    """
    Create and display PCA variance analysis plot.

    Args:
        df_scaled (pd.DataFrame): Scaled input DataFrame
        variance_threshold (float, optional): Variance threshold. Defaults to 0.90.

    Returns:
        int: Number of components needed for threshold
    """
    # Perform PCA
    pca, _ = perform_pca(df_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    individual_variance = pca.explained_variance_ratio_

    # Find components needed for threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Create plot
    fig, ax1, ax2 = setup_plot()
    plot_cumulative_variance(ax1, cumulative_variance, n_components, variance_threshold)
    plot_individual_variance(ax2, individual_variance)
    customize_axes(ax1, ax2, cumulative_variance, individual_variance)

    # Add summary and save
    print_summary(individual_variance, cumulative_variance, n_components, variance_threshold)
    plt.tight_layout()
    save_plot()
    plt.show()

    return n_components


if __name__ == '__main__':
    # Assuming df_scaled is already loaded
    n_components = plot_pca_cumulative_variance(df_scaled, variance_threshold=0.90)