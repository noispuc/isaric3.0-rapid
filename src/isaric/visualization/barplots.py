import matplotlib.pyplot as plt

def plot_bar(df, x_col, y_col, title="Bar Plot"):
    """
    Description:
        Generates a bar plot for comparing categorical values.

    Args:
        df (pandas.DataFrame): Input dataset.
        x_col (str): Column name for categories.
        y_col (str): Column name for values.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: Bar plot figure.
    """
    fig, ax = plt.subplots()
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    return fig
