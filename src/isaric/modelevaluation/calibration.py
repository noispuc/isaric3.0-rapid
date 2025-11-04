from sklearn.calibration import calibration_curve
import pandas as pd

def compute_calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):
    """
    Description:
        Computes calibration curve data for binary classification models.

    Args:
        y_true (array-like): True binary labels.
        y_prob (array-like): Predicted probabilities.
        n_bins (int): Number of bins to discretize the [0,1] interval.
        strategy (str): Binning strategy ('uniform' or 'quantile').

    Returns:
        pandas.DataFrame: DataFrame with mean predicted and observed probabilities per bin.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"predicted": prob_pred, "observed": prob_true})
