import matplotlib.pyplot as plt

def plot_forest(coefficients, ci_lower, ci_upper, labels, title="Forest Plot"):
    """
    Description:
        Generates a forest plot to visualize effect sizes and confidence intervals.

    Args:
        coefficients (list): List of point estimates.
        ci_lower (list): List of lower bounds of confidence intervals.
        ci_upper (list): List of upper bounds of confidence intervals.
        labels (list): List of variable names.
        title (str): Title of the plot.

    Returns:
        matplotlib.figure.Figure: Forest plot figure.
    """
    fig, ax = plt.subplots()
    ax.errorbar(coefficients, range(len(coefficients)), xerr=[coefficients - ci_lower, ci_upper - coefficients], fmt='o')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.invert_yaxis()
    return fig
