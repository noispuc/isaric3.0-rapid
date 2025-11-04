from upsetplot import UpSet
import matplotlib.pyplot as plt

def plot_upset(data, subset_column="sets"):
    """
    Description:
        Generates an UpSet plot to visualize set intersections.

    Args:
        data (pandas.Series): Series with multi-index representing set combinations.
        subset_column (str): Label for the subset axis.

    Returns:
        matplotlib.figure.Figure: UpSet plot figure.
    """
    upset = UpSet(data, subset_name=subset_column)
    fig = plt.figure()
    upset.plot(fig=fig)
    return fig
