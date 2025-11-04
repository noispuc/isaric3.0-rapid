import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

def plot_sankey(flows, labels, orientations, title="Sankey Diagram"):
    """
    Description:
        Generates a Sankey diagram to visualize flow distributions.

    Args:
        flows (list): List of flow values (positive for inputs, negative for outputs).
        labels (list): List of labels for each flow.
        orientations (list): List of orientations ('left', 'right', 'up', 'down').

    Returns:
        matplotlib.figure.Figure: Sankey diagram figure.
    """
    fig, ax = plt.subplots()
    sankey = Sankey(ax=ax, unit=None)
    sankey.add(flows=flows, labels=labels, orientations=orientations)
    sankey.finish()
    ax.set_title(title)
    return fig
