def save_plot(fig, filename):
    """
    Description:
        Saves a matplotlib figure to disk.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filename (str): Path to save the image file.

    Returns:
        None
    """
    fig.savefig(filename)
