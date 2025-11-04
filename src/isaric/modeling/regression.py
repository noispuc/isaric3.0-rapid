from sklearn.linear_model import LinearRegression

def fit_linear_regression(df, target):
    """
    Description:
        Fits a linear regression model to predict a continuous target variable.

    Args:
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.

    Returns:
        sklearn.linear_model.LinearRegression: Trained regression model.
    """
    X = df.drop(columns=[target])
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    return model
