from sklearn.linear_model import Lasso

def select_features_lasso(df, target, alpha=0.01):
    """
    Description:
        Selects features using Lasso regularization.

    Args:
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.
        alpha (float): Regularization strength.

    Returns:
        list: List of selected feature names.
    """
    X = df.drop(columns=[target])
    y = df[target]
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return list(X.columns[model.coef_ != 0])
