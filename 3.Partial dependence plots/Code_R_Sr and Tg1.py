#!/usr/bin/env python3
"""
Visualization script for polyurethane lignin data analysis.

Creates joint plots showing relationships between Swelling ratio and Tg,
with additional visualization of lignin weight percentage.
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

    # Filter data based on isocyanate type
    filtered_df = df[df['Isocyanate type'] != (1 - isocyanate_filter)]
    
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


def customize_joint_plot(joint_plot, df_numeric):
    """
    Customize the appearance of the joint plot.

    Args:
        joint_plot (seaborn.JointGrid): The joint plot to customize
        df_numeric (pd.DataFrame): DataFrame with the plot data

    Returns:
        seaborn.JointGrid: The customized joint plot
    """
    # Add scatter plot with lignin percentage coloring
    sns.scatterplot(
        data=df_numeric,
        x='Swelling ratio (%)',
        y='Tg (°C)',
        hue='Lignin (wt%)',
        palette='viridis',
        ax=joint_plot.ax_joint,
        alpha=0.7
    )

    # Add KDE plot
    joint_plot.plot_joint(
        sns.kdeplot,
        levels=5,
        color="r",
        zorder=0,
        alpha=0.5
    )

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

    # Add and customize legend
    plt.legend(
        title=' Lignin (wt %)',
        title_fontsize='16',
        fontsize='16',
        loc='upper right'
    )

    return joint_plot


def save_plot(joint_plot, base_filename):
    """
    Save the plot in multiple formats.

    Args:
        joint_plot (seaborn.JointGrid): The plot to save
        base_filename (str): Base name for the output files
    """
    formats = ['png', 'tiff', 'pdf', 'eps', 'svg']
    for fmt in formats:
        dpi = 600 if fmt in ['png', 'tiff'] else None
        joint_plot.savefig(
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
    # Prepare data
    df_numeric = prepare_data(df, isocyanate_filter=1)  # 1 for HDIt

    # Create and customize plot
    joint_plot = create_joint_plot(df_numeric)
    joint_plot = customize_joint_plot(joint_plot, df_numeric)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Uncomment to save the plot
    # save_plot(joint_plot, 'R_Sr and Tg_HDIt')


if __name__ == '__main__':
    # Assuming df is already loaded
    main(df)