def summarize_validation_scores(scores):
    """
    Description:
        Summarizes validation scores with mean and standard deviation.

    Args:
        scores (list): List of numeric scores.

    Returns:
        dict: Dictionary with mean and standard deviation.
    """
    import numpy as np
    return {
        "mean": np.mean(scores),
        "std": np.std(scores)
    }
