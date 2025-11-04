from sklearn.model_selection import train_test_split

def train_test_split_evaluation(df, target, test_size=0.2, random_state=42):
    """
    Description:
        Splits the dataset into training and testing sets for evaluation purposes.

    Args:
        df (pandas.DataFrame): Input dataset.
        target (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
