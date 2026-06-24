def save_plot(fig, filename):
    """
    Saves a matplotlib figure to disk.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        filename (str): Path to save the image file.
    """
    fig.savefig(filename)


def save_plotly(fig, filename, format="html"):
    """
    Saves a Plotly figure to disk.

    Args:
        fig (plotly.graph_objs.Figure): Figure to save.
        filename (str): Output path (without extension if format != 'html').
        format (str): 'html' for interactive, or image format ('png', 'svg', 'pdf').
    """
    if format == "html":
        fig.write_html(filename if filename.endswith(".html") else f"{filename}.html")
    else:
        fig.write_image(filename if "." in filename else f"{filename}.{format}")
