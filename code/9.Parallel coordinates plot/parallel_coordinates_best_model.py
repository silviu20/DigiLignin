# -*- coding: utf-8 -*-
"""
Parallel Coordinates Plot for Best Model (5 features)
Interactive visualization showing relationships between input features and predicted Tg
Features: Lignin (wt%), Co-polyol type (PTHF), r, Copolyol (wt%), Isocyanate (wt%)
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import os

def create_parallel_coordinates_plot(df):
    """
    Create a parallel coordinates plot from the given DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing experimental data
    
    Returns:
        plotly.graph_objects.Figure: The parallel coordinates plot
    """
    return go.Figure(data=go.Parcoords(
        line=dict(
            color=df['Predicted_Tg'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text='Tg / °C', font=dict(size=20)),
                tickvals=[-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                ticktext=[f'{temp}°C' for temp in [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]],
                ticks="outside",
                tickfont=dict(size=18),
            )
        ),
        dimensions=[
            dict(
                range=[df['Lignin (wt%)'].min(), df['Lignin (wt%)'].max()],
                label='<b>Lignin (wt%)</b>',
                values=df['Lignin (wt%)'],
                tickvals=list(range(0, 101, 10))
            ),
            dict(
                range=[
                    df['Co-polyol type (PTHF)'].min(),
                    df['Co-polyol type (PTHF)'].max()
                ],
                label='<b>Co-polyol type (PTHF)</b>',
                values=df['Co-polyol type (PTHF)'],
                tickvals=[250, 650, 1000]
            ),
            dict(
                range=[df['r'].min(), df['r'].max()],
                label='<b>r (Ratio)</b>',
                values=df['r'],
                tickvals=[0.6, 0.8, 1.0, 1.2, 1.4]
            ),
            dict(
                range=[df['Copolyol (wt%)'].min(), df['Copolyol (wt%)'].max()],
                label='<b>Copolyol (wt%)</b>',
                values=df['Copolyol (wt%)'],
                tickvals=list(range(0, 101, 10))
            ),
            dict(
                range=[
                    df['Isocyanate (wt%)'].min(),
                    df['Isocyanate (wt%)'].max()
                ],
                label='<b>Isocyanate (wt%)</b>',
                values=df['Isocyanate (wt%)'],
                tickvals=[0, 5, 10, 15, 20]
            ),
            dict(
                range=[df['Predicted_Tg'].min(), df['Predicted_Tg'].max()],
                label='<b>Predicted Tg / °C</b>',
                values=df['Predicted_Tg'],
                tickvals=list(range(-20, 101, 10))
            )
        ]
    ))

def main():
    """Main function to execute the script."""
    print("="*80)
    print("PARALLEL COORDINATES PLOT - BEST MODEL")
    print("="*80)
    print("Creating interactive visualization of feature relationships")
    print("="*80)
    
    # Set the default renderer to open the plot in a browser
    pio.renderers.default = "browser"
    
    # Load the data from CSV file
    csv_filename = '../8.Extrapolation/closest_inputs_best_model.csv'
    
    print(f"\nLoading data from {csv_filename}...")
    if not os.path.exists(csv_filename):
        print(f"❌ ERROR: File not found: {csv_filename}")
        print("Please run '../8.Extrapolation/adaptive_grid_search_best_model.py' first.")
        return
    
    df = pd.read_csv(csv_filename)
    print(f"Loaded {len(df)} data points")
    
    # Display data summary
    print("\nData summary:")
    print(f"  Lignin (wt%): {df['Lignin (wt%)'].min():.1f} - {df['Lignin (wt%)'].max():.1f}")
    print(f"  Co-polyol type (PTHF): {df['Co-polyol type (PTHF)'].unique()}")
    print(f"  r: {df['r'].min():.2f} - {df['r'].max():.2f}")
    print(f"  Copolyol (wt%): {df['Copolyol (wt%)'].min():.1f} - {df['Copolyol (wt%)'].max():.1f}")
    print(f"  Isocyanate (wt%): {df['Isocyanate (wt%)'].min():.1f} - {df['Isocyanate (wt%)'].max():.1f}")
    print(f"  Predicted Tg: {df['Predicted_Tg'].min():.1f}°C - {df['Predicted_Tg'].max():.1f}°C")
    
    # Create the parallel coordinates plot
    print("\nCreating parallel coordinates plot...")
    parallel_coords = create_parallel_coordinates_plot(df)
    
    # Update the layout of the plot
    parallel_coords.update_layout(
        plot_bgcolor='white',
        font=dict(size=22, color='black'),
        hoverlabel=dict(font_size=20),
        title=dict(
            text='Parallel Coordinates Plot - Best Model (5 Features)',
            font=dict(size=24),
            x=0.5,
            xanchor='center'
        )
    )
    
    # Save as HTML
    html_filename = 'parallel_coordinates_best_model.html'
    parallel_coords.write_html(html_filename)
    print(f"  ✓ Saved interactive plot to {html_filename}")
    
    # Display the plot in the browser
    print("\nOpening plot in browser...")
    parallel_coords.show()
    
    print("\n" + "="*80)
    print("PARALLEL COORDINATES PLOT COMPLETED!")
    print("="*80)
    print("\nInteractive features:")
    print("  - Click and drag on axes to filter data")
    print("  - Hover over lines to see values")
    print("  - Use colorbar to identify Tg values")
    print("="*80)

if __name__ == '__main__':
    main()
