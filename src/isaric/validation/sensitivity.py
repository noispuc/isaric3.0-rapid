def sensitivity_analysis(model, X, y, perturbation=0.01):
    """
    Description:
        Evaluates model sensitivity by perturbing input features slightly.

    Args:
        model (sklearn estimator): Trained model.
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target vector.
        perturbation (float): Magnitude of feature perturbation.

    Returns:
        dict: Dictionary with original and perturbed accuracy scores.
    """
    from sklearn.metrics import accuracy_score
    y_pred_original = model.predict(X)
    original_score = accuracy_score(y, y_pred_original)

    X_perturbed = X + perturbation
    y_pred_perturbed = model.predict(X_perturbed)
    perturbed_score = accuracy_score(y, y_pred_perturbed)

    return {
        "original_accuracy": original_score,
        "perturbed_accuracy": perturbed_score
    }
