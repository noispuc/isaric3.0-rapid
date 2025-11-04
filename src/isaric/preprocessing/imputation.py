from sklearn.impute import SimpleImputer

def impute_missing(df, strategy="mean"):
    """
    Description:
        Imputes missing values using a specified strategy.

    Args:
        df (pandas.DataFrame): Input dataset.
        strategy (str): Imputation strategy ('mean', 'median', 'most_frequent').

    Returns:
        pandas.DataFrame: Dataset with imputed values.
    """
    imputer = SimpleImputer(strategy=strategy)
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
