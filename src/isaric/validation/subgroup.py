def subgroup_analysis(model, df, target, subgroup_column):
    """
    Description:
        Evaluates model performance across different subgroups in the dataset.

    Args:
        model (sklearn estimator): Trained model.
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.
        subgroup_column (str): Column used to define subgroups.

    Returns:
        dict: Dictionary with accuracy per subgroup.
    """
    from sklearn.metrics import accuracy_score
    results = {}
    for group in df[subgroup_column].unique():
        subset = df[df[subgroup_column] == group]
        X = subset.drop(columns=[target])
        y = subset[target]
        y_pred = model.predict(X)
        results[group] = accuracy_score(y, y_pred)
    return results
