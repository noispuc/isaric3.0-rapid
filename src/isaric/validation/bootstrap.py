import numpy as np

def bootstrap_validation(model, X, y, n_iterations=1000):
    """
    Description:
        Performs bootstrap resampling to estimate model performance variability.

    Args:
        model (sklearn estimator): Trained model.
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target vector.
        n_iterations (int): Number of bootstrap samples.

    Returns:
        list: List of accuracy scores across bootstrap samples.
    """
    from sklearn.metrics import accuracy_score
    scores = []
    for _ in range(n_iterations):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
        y_pred = model.predict(X_sample)
        scores.append(accuracy_score(y_sample, y_pred))
    return scores
