# Standard library imports

# Third-party imports
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


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
                title=dict(text='Tg', font=dict(size=20)),
                tickvals=[-10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                ticktext=[f'{temp}°C' for temp in range(-10, 101, 10)],
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
                range=[df['Ratio'].min(), df['Ratio'].max()],
                label='<b>Ratio</b>',
                values=df['Ratio'],
                tickvals=[0.6, 0.8, 1.0, 1.2, 1.4]
            ),
            dict(
                range=[df['Co-polyol (wt%)'].min(), df['Co-polyol (wt%)'].max()],
                label='<b>Co-polyol (wt%)</b>',
                values=df['Co-polyol (wt%)'],
                tickvals=list(range(0, 101, 10))
            ),
            dict(
                range=[
                    df['Isocyanate (mmol NCO)'].min(),
                    df['Isocyanate (mmol NCO)'].max()
                ],
                label='<b>Isocyanate (mmol NCO)</b>',
                values=df['Isocyanate (mmol NCO)'],
                tickvals=[0, 5, 10, 15, 20]
            ),
            dict(
                range=[df['Tin(II) octoate'].min(), df['Tin(II) octoate'].max()],
                label='<b>Tin(II) octoate</b>',
                values=df['Tin(II) octoate'],
                tickvals=[0, 0.6, 1.3, 2]
            ),
            dict(
                range=[df['Predicted_Tg'].min(), df['Predicted_Tg'].max()],
                label='<b>Predicted Tg</b>',
                values=df['Predicted_Tg'],
                tickvals=[-10] + list(range(0, 91, 10)) + [93]
            )
        ]
    ))


def main():
    """Main function to execute the script."""
    # Set the default renderer to open the plot in a browser
    pio.renderers.default = "browser"

    # Load the data from a CSV file
    file_path = ('C:/Users/P70090917/Desktop/Polyuerthane Lignin/Experiments/'
                 'dataset2/rework 21_Nov_2024/testing fesature combinations/'
                 'rapid adaptive/closest_inputs_test_results.csv')
    df = pd.read_csv(file_path)

    # Create the parallel coordinates plot
    parallel_coords = create_parallel_coordinates_plot(df)

    # Update the layout of the plot
    parallel_coords.update_layout(
        plot_bgcolor='white',
        font=dict(size=22, color='black'),
        hoverlabel=dict(font_size=20)
    )

    # Display the plot in the browser
    parallel_coords.show()


if __name__ == '__main__':
    main()