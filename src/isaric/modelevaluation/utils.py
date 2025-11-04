def summarize_scores(scores):
    """
    Description:
        Summarizes cross-validation scores with mean and standard deviation.

    Args:
        scores (array-like): List or array of numeric scores.

    Returns:
        dict: Dictionary with mean and standard deviation of scores.
    """
    import numpy as np
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores)
    }
