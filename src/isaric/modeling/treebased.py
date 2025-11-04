from sklearn.ensemble import RandomForestClassifier

def fit_random_forest(df, target, n_estimators=100):
    """
    Description:
        Fits a Random Forest classifier to predict a categorical target variable.

    Args:
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.
        n_estimators (int): Number of trees in the forest.

    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained Random Forest model.
    """
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model
