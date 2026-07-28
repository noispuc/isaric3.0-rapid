from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

def perform_cross_validation(model, X, y, cv=5, scoring="accuracy"):
    """
    Description:
        Performs k-fold cross-validation on a given model.

    Args:
        model (sklearn estimator): Trained model or pipeline.
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): Target vector.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric to evaluate performance.

    Returns:
        numpy.ndarray: Array of cross-validation scores.
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring)


def build_repeated_stratified_kfold(n_splits=5, n_repeats=10, random_state=42):
    """
    Description:
        Cria um esquema de k-fold estratificado repetido, usado para tuning
        de hiperparâmetros dentro do bloco de treino (nunca sobre o teste
        temporal), reduzindo a variância da estimativa de performance.

    Args:
        n_splits (int): Número de folds por repetição.
        n_repeats (int): Número de repetições do k-fold.
        random_state (int): Semente para reprodutibilidade.

    Returns:
        sklearn.model_selection.RepeatedStratifiedKFold: Esquema de CV pronto
            para ser passado a GridSearchCV/RandomizedSearchCV.
    """
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
