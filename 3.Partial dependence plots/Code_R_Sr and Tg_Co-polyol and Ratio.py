#!/usr/bin/env python3
"""
Visualization script for polyurethane lignin data analysis.

Creates joint plots showing relationships between Swelling ratio and Tg,
with additional visualization of co-polyol type and ratio parameters.

Created on Sat Oct 12 18:22:41 2024
Author: P70090917
"""

# Standard library imports

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def prepare_data(df, isocyanate_filter=1):
    """
    Prepare and filter the dataset for visualization.

    Args:
        df (pd.DataFrame): Input DataFrame
        isocyanate_filter (int): Value to filter Isocyanate type (0 for HDI, 1 for HDIt)

    Returns:
        pd.DataFrame: Processed DataFrame with numeric values
    """
    columns_to_plot = [
        'Co-polyol type (PTHF)',
        'Ratio',
        'Tg (°C)',
        'Swelling ratio (%)'
    ]

    # Filter data based on isocyanate type
    filtered_df = df[df['Isocyanate type'] != isocyanate_filter]
    
    # Convert to numeric values
    return filtered_df[columns_to_plot].apply(
        lambda x: pd.to_numeric(x, errors='coerce')
    )


def create_joint_plot(df_numeric):
    """
    Create a joint plot showing the relationship between Swelling ratio and Tg.

    Args:
        df_numeric (pd.DataFrame): DataFrame with numeric values

    Returns:
        seaborn.JointGrid: The created joint plot
    """
    # Set style parameters
    sns.set_style("whitegrid")
    sns.set_palette("deep")

    # Create base joint plot
    joint_plot = sns.jointplot(
        data=df_numeric,
        x='Swelling ratio (%)',
        y='Tg (°C)',
        kind='scatter',
        height=8,
        ratio=7,
        space=0.2,
        color='#118ce8',
        marginal_kws=dict(bins=25, fill=True)
    )

    return joint_plot


def add_styled_scatter(joint_plot, df_numeric):
    """
    Add a styled scatter plot with ratio and co-polyol type visualization.

    Args:
        joint_plot (seaborn.JointGrid): Base joint plot
        df_numeric (pd.DataFrame): Data for plotting

    Returns:
        seaborn.JointGrid: Updated joint plot
    """
    sns.scatterplot(
        data=df_numeric,
        x='Swelling ratio (%)',
        y='Tg (°C)',
        hue='Ratio',
        style='Co-polyol type (PTHF)',
        palette='viridis',
        ax=joint_plot.ax_joint,
        alpha=0.7,
        s=100
    )

    # Add KDE plot
    joint_plot.plot_joint(
        sns.kdeplot,
        levels=5,
        color="r",
        zorder=0,
        alpha=0.5
    )

    return joint_plot


def customize_plot(joint_plot, df_numeric):
    """
    Customize the appearance of the joint plot.

    Args:
        joint_plot (seaborn.JointGrid): The joint plot to customize
        df_numeric (pd.DataFrame): DataFrame with the plot data

    Returns:
        seaborn.JointGrid: The customized joint plot
    """
    # Set labels and style
    joint_plot.set_axis_labels(
        'Swelling ratio (%)',
        'Tg (°C)',
        fontsize=22
    )
    
    # Customize plot appearance
    joint_plot.ax_joint.collections[0].set_alpha(0.5)
    joint_plot.ax_joint.set_facecolor('white')

    # Add correlation statistics
    correlation = df_numeric['Swelling ratio (%)'].corr(df_numeric['Tg (°C)'])
    stats_text = f"Correlation: {correlation:.2f}"
    joint_plot.ax_joint.text(
        0.05, 0.95,
        stats_text,
        transform=joint_plot.ax_joint.transAxes,
        verticalalignment='top',
        fontsize=16,
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.7
        )
    )

    # Adjust tick parameters
    joint_plot.ax_joint.tick_params(
        axis='both',
        which='major',
        labelsize=20
    )

    # Add legend
    plt.legend(fontsize='16', loc='upper right')

    # Add color normalization for ratio
    norm = plt.Normalize(
        df_numeric['Ratio'].min(),
        df_numeric['Ratio'].max()
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    return joint_plot


def save_plot(base_filename, formats=None):
    """
    Save the plot in multiple formats.

    Args:
        base_filename (str): Base name for the output files
        formats (list): List of formats to save (default: png, tiff, pdf, svg)
    """
    if formats is None:
        formats = ['png', 'tiff', 'pdf', 'svg']

    for fmt in formats:
        dpi = 600 if fmt in ['png', 'tiff'] else None
        plt.savefig(
            f'{base_filename}.{fmt}',
            dpi=dpi,
            bbox_inches='tight'
        )


def main(df):
    """
    Main function to create and display the visualization.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data
    """
    # Prepare data (filtering for HDI)
    df_numeric = prepare_data(df, isocyanate_filter=1)

    # Create base plot
    joint_plot = create_joint_plot(df_numeric)

    # Add styled scatter plot
    joint_plot = add_styled_scatter(joint_plot, df_numeric)

    # Customize plot appearance
    joint_plot = customize_plot(joint_plot, df_numeric)

    # Adjust layout
    plt.tight_layout()

    # Save plot in various formats
    save_plot('R_Sr and Tg_Co-polyol and Ratio_HDI')

    # Display plot
    plt.show()


if __name__ == '__main__':
    # Assuming df is already loaded
    main(df)