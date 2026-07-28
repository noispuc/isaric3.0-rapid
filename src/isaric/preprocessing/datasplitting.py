from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.2, random_state=42):
    """
    Description:
        Splits the dataset into training and testing sets.

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


def temporal_train_test_split(df, year_column, train_end_year, test_start_year):
    """
    Description:
        Divide o dataset em treino e teste com base em um corte temporal
        (ano), garantindo que o treino contenha apenas anos anteriores ao
        teste e evitando vazamento de informação futura.

    Args:
        df (pandas.DataFrame): Dataset de entrada, contendo year_column.
        year_column (str): Nome da coluna com o ano de referência (ex.: ano
            do sintoma primário).
        train_end_year (int): Último ano (inclusive) incluído no treino.
        test_start_year (int): Primeiro ano (inclusive) incluído no teste.
            Deve ser maior que train_end_year; anos entre os dois cortes
            (exclusive) ficam de fora de ambos os conjuntos.

    Returns:
        tuple: (df_train, df_test), pandas.DataFrame com todas as colunas
            originais preservadas.
    """
    if year_column not in df.columns:
        raise ValueError(f"Coluna de ano '{year_column}' não encontrada no dataset.")

    if test_start_year <= train_end_year:
        raise ValueError(
            "test_start_year deve ser maior que train_end_year para evitar "
            "vazamento de informação temporal entre treino e teste."
        )

    df_train = df[df[year_column] <= train_end_year].copy()
    df_test = df[df[year_column] >= test_start_year].copy()

    if df_train.empty:
        raise ValueError(f"Nenhuma linha encontrada para o treino (ano <= {train_end_year}).")
    if df_test.empty:
        raise ValueError(f"Nenhuma linha encontrada para o teste (ano >= {test_start_year}).")

    return df_train, df_test
