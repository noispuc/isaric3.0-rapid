import matplotlib.pyplot as plt

def plot_line(df, x_col, y_col, title="Line Plot"):
    """
    Description:
        Generates a line plot for visualizing trends over time or ordered categories.

    Args:
        df (pandas.DataFrame): Input dataset.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: Line plot figure.
    """
    fig, ax = plt.subplots()
    ax.plot(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig
