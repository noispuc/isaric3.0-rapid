import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(df, title="Heatmap"):
    """
    Description:
        Generates a heatmap to visualize correlations or intensity across a matrix.

    Args:
        df (pandas.DataFrame): Input dataset (typically a correlation matrix).
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: Heatmap figure.
    """
    fig, ax = plt.subplots()
    sns.heatmap(df, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title(title)
    return fig
