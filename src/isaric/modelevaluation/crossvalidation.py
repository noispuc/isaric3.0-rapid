from sklearn.model_selection import cross_val_score

def perform_cross_validation(model, X, y, cv=5, scoring="accuracy"):
    """
    Description:
        Performs k-fold cross-validation on a given model.

    Args:
        model (sklearn estimator): Trained model or pipeline.
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target vector.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric to evaluate performance.

    Returns:
        numpy.ndarray: Array of cross-validation scores.
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)
