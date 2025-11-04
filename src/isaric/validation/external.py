def validate_with_external_dataset(model, external_df, target):
    """
    Description:
        Validates a trained model using an independent external dataset.

    Args:
        model (sklearn estimator): Trained model.
        external_df (pandas.DataFrame): External dataset for validation.
        target (str): Name of the target column in the external dataset.

    Returns:
        dict: Dictionary with performance metrics on the external dataset.
    """
    from sklearn.metrics import accuracy_score
    X_ext = external_df.drop(columns=[target])
    y_ext = external_df[target]
    y_pred = model.predict(X_ext)
    return {"accuracy": accuracy_score(y_ext, y_pred)}
