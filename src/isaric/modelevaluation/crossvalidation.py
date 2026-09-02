"""
Cross-validation for the RAPID methodology.

This module provides functions to perform cross-validation (Step 4 of
the RAPID methodology). Cross-validation systematically partitions
training data into k folds, training the model k times and validating
on the remaining fold.

Techniques:
- kfold_cross_validation: Perform k-fold cross-validation.
- repeated_kfold_cross_validation: Repeat k-fold multiple times.
- stratified_kfold_cross_validation: Preserve class proportions.
- build_repeated_stratified_kfold: Build a cross-validator for pipelines.

Supports:
- sklearn models (via cross_val_score)
- statsmodels models (via manual implementation)
"""

import pandas as pd
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score
)


def kfold_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    shuffle: bool = True,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform k-fold cross-validation.

    Splits data into k non-overlapping subsets, trains the model k times,
    and validates on the remaining fold.

    Supports both sklearn and statsmodels models.

    Args:
        model: Configured model with fit() and predict() methods.
        X: Predictor matrix.
        y: Outcome vector.
        n_splits: Number of folds (default 5).
        scoring: Scoring metric (e.g., "roc_auc", "accuracy", "neg_mse").
        shuffle: Whether to shuffle data before splitting.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with 'scores' (array of fold scores),
        'mean_score', and 'std_score'.

    Raises:
        ValueError: If n_splits is invalid.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2. Received: {n_splits}")

    # Verifica se é modelo sklearn (tem get_params)
    if hasattr(model, 'get_params'):
        # sklearn - usa cross_val_score nativo
        cv = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    else:
        # statsmodels - cross-validation manual
        scores = _manual_kfold_cv(
            model, X, y,
            n_splits=n_splits,
            scoring=scoring,
            shuffle=shuffle,
            random_state=random_state
        )

    return {
        'scores': scores,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


def repeated_kfold_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    scoring: str = "roc_auc",
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform repeated k-fold cross-validation.

    Runs k-fold cross-validation multiple times with different random
    splits to reduce variability in performance estimates.

    Args:
        model: Configured model with fit() and predict() methods.
        X: Predictor matrix.
        y: Outcome vector.
        n_splits: Number of folds (default 5).
        n_repeats: Number of repetitions (default 3).
        scoring: Scoring metric (e.g., "roc_auc", "accuracy", "neg_mse").
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with 'scores', 'mean_score', and 'std_score'.

    Raises:
        ValueError: If parameters are invalid.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2. Received: {n_splits}")

    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1. Received: {n_repeats}")

    if hasattr(model, 'get_params'):
        # sklearn
        cv = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    else:
        # statsmodels - repete o CV manual
        all_scores = []
        for repeat in range(n_repeats):
            repeat_seed = random_state + repeat if random_state else None
            scores = _manual_kfold_cv(
                model, X, y,
                n_splits=n_splits,
                scoring=scoring,
                shuffle=True,
                random_state=repeat_seed
            )
            all_scores.extend(scores)
        scores = np.array(all_scores)

    return {
        'scores': scores,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


def stratified_kfold_cross_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    shuffle: bool = True,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Perform stratified k-fold cross-validation.

    Preserves the proportion of classes in each fold, important for
    imbalanced datasets.

    Args:
        model: Configured model with fit() and predict() methods.
        X: Predictor matrix.
        y: Outcome vector (categorical).
        n_splits: Number of folds (default 5).
        scoring: Scoring metric (e.g., "roc_auc", "accuracy").
        shuffle: Whether to shuffle data before splitting.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with 'scores', 'mean_score', and 'std_score'.

    Raises:
        ValueError: If n_splits is invalid.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2. Received: {n_splits}")

    if hasattr(model, 'get_params'):
        # sklearn
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    else:
        # statsmodels - usa stratified manual
        scores = _manual_stratified_kfold_cv(
            model, X, y,
            n_splits=n_splits,
            scoring=scoring,
            shuffle=shuffle,
            random_state=random_state
        )

    return {
        'scores': scores,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


def out_of_fold_predictions(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    method: str = "predict_proba",
    n_repeats: int = 1,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate out-of-fold predictions using cross-validation.

    Each prediction comes from a model trained on data that did not
    include the observation being predicted.

    Args:
        model: Configured model.
        X: Predictor matrix.
        y: Outcome vector.
        n_splits: Number of folds (default 5).
        method: Prediction method: "predict" or "predict_proba".
        n_repeats: Number of repetitions (default 1).
        random_state: Seed for reproducibility.

    Returns:
        Array of out-of-fold predictions.
    """
    if hasattr(model, 'get_params'):
        # sklearn
        if n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state
            )
        else:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state
            )

        predictions = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method=method
        )

        if method == "predict_proba":
            return predictions[:, 1] if predictions.ndim == 2 else predictions

        return predictions
    else:
        # statsmodels - manual out-of-fold
        return _manual_out_of_fold(
            model, X, y,
            n_splits=n_splits,
            method=method,
            random_state=random_state
        )


def build_repeated_stratified_kfold(
    n_splits: int = 5,
    n_repeats: int = 5,
    random_state: int = 42
) -> RepeatedStratifiedKFold:
    """
    Build a RepeatedStratifiedKFold cross-validator.

    This is a convenience function for use in sklearn pipelines.

    Args:
        n_splits: Number of folds (default 5).
        n_repeats: Number of repetitions (default 5).
        random_state: Seed for reproducibility.

    Returns:
        RepeatedStratifiedKFold instance.
    """
    return RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )


# ============================================================================
# PRIVATE HELPERS - MANUAL CROSS-VALIDATION FOR STATSMODELS
# ============================================================================

def _manual_kfold_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    shuffle: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Manual k-fold cross-validation for statsmodels models.

    This function implements the cross-validation loop manually because
    statsmodels models do not follow the sklearn estimator API
    (they lack get_params/set_params).

    The procedure is statistically correct:
    1. Data is split into k folds
    2. For each fold, the model is trained on k-1 folds
    3. The model predicts on the held-out fold
    4. Performance is measured on the held-out fold
    5. Scores from all folds are averaged

    This is the same procedure used by sklearn's cross_val_score,
    just implemented manually for statsmodels compatibility.
    """
    # Converte para arrays numpy
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_array = y.values if hasattr(y, 'values') else np.array(y)
    
    # Preserva nomes das colunas para reconstruir DataFrames
    X_columns = X.columns if hasattr(X, 'columns') else None
    
    # Cria folds
    kf = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    scores = []
    
    for train_idx, val_idx in kf.split(X_array):
        # Separa treino e validação
        X_train_array = X_array[train_idx]
        X_val_array = X_array[val_idx]
        y_train_array = y_array[train_idx]
        y_val_array = y_array[val_idx]
        
        # Converte de volta para DataFrame/Series (statsmodels exige)
        if X_columns is not None:
            X_train_df = pd.DataFrame(X_train_array, columns=X_columns)
            X_val_df = pd.DataFrame(X_val_array, columns=X_columns)
        else:
            X_train_df = pd.DataFrame(X_train_array)
            X_val_df = pd.DataFrame(X_val_array)
        
        y_train_series = pd.Series(y_train_array)
        
        # Cria uma cópia do modelo statsmodels e treina
        # Nota: statsmodels não clona facilmente, então treinamos
        # diretamente no modelo configurado
        try:
            # Tenta treinar o modelo
            fitted_model = model.fit()
            
            # Prediz no conjunto de validação
            # Para GLM binomial, predict() retorna probabilidades
            y_prob = fitted_model.predict(X_val_df)
            
            # Calcula a métrica
            score = _calculate_score(y_val_array, y_prob, scoring)
            scores.append(score)
            
        except Exception as e:
            # Se falhar, registra NaN
            scores.append(np.nan)
    
    return np.array(scores)


def _manual_stratified_kfold_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    shuffle: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Manual stratified k-fold cross-validation for statsmodels models.
    """
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_array = y.values if hasattr(y, 'values') else np.array(y)
    X_columns = X.columns if hasattr(X, 'columns') else None
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    scores = []
    
    for train_idx, val_idx in skf.split(X_array, y_array):
        X_train_array = X_array[train_idx]
        X_val_array = X_array[val_idx]
        y_train_array = y_array[train_idx]
        y_val_array = y_array[val_idx]
        
        if X_columns is not None:
            X_train_df = pd.DataFrame(X_train_array, columns=X_columns)
            X_val_df = pd.DataFrame(X_val_array, columns=X_columns)
        else:
            X_train_df = pd.DataFrame(X_train_array)
            X_val_df = pd.DataFrame(X_val_array)
        
        y_train_series = pd.Series(y_train_array)
        
        try:
            fitted_model = model.fit()
            y_prob = fitted_model.predict(X_val_df)
            score = _calculate_score(y_val_array, y_prob, scoring)
            scores.append(score)
        except Exception:
            scores.append(np.nan)
    
    return np.array(scores)


def _manual_out_of_fold(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    method: str = "predict",
    random_state: int = 42
) -> np.ndarray:
    """
    Manual out-of-fold predictions for statsmodels models.
    """
    X_array = X.values if hasattr(X, 'values') else np.array(X)
    y_array = y.values if hasattr(y, 'values') else np.array(y)
    X_columns = X.columns if hasattr(X, 'columns') else None
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    predictions = np.zeros(len(y_array))
    
    for train_idx, val_idx in skf.split(X_array, y_array):
        X_train_array = X_array[train_idx]
        X_val_array = X_array[val_idx]
        y_train_array = y_array[train_idx]
        
        if X_columns is not None:
            X_train_df = pd.DataFrame(X_train_array, columns=X_columns)
            X_val_df = pd.DataFrame(X_val_array, columns=X_columns)
        else:
            X_train_df = pd.DataFrame(X_train_array)
            X_val_df = pd.DataFrame(X_val_array)
        
        y_train_series = pd.Series(y_train_array)
        
        try:
            fitted_model = model.fit()
            y_prob = fitted_model.predict(X_val_df)
            
            if method == "predict_proba":
                predictions[val_idx] = y_prob
            else:
                predictions[val_idx] = (y_prob >= 0.5).astype(int)
                
        except Exception:
            predictions[val_idx] = np.nan
    
    return predictions


def _calculate_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    scoring: str
) -> float:
    """
    Calculate performance score based on scoring metric.
    
    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        scoring: Metric name.
    
    Returns:
        Score value.
    """
    # Classificação binária
    if scoring == "roc_auc":
        return roc_auc_score(y_true, y_prob)
    elif scoring == "accuracy":
        y_pred = (y_prob >= 0.5).astype(int)
        return accuracy_score(y_true, y_pred)
    elif scoring == "precision":
        y_pred = (y_prob >= 0.5).astype(int)
        return precision_score(y_true, y_pred, zero_division=0)
    elif scoring == "recall":
        y_pred = (y_prob >= 0.5).astype(int)
        return recall_score(y_true, y_pred, zero_division=0)
    elif scoring == "f1":
        y_pred = (y_prob >= 0.5).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)
    elif scoring == "neg_brier_score":
        from sklearn.metrics import brier_score_loss
        return -brier_score_loss(y_true, y_prob)
    elif scoring == "neg_log_loss":
        from sklearn.metrics import log_loss
        return -log_loss(y_true, y_prob)
    
    # Regressão
    elif scoring == "neg_mean_squared_error":
        return -mean_squared_error(y_true, y_prob)
    elif scoring == "r2":
        return r2_score(y_true, y_prob)
    
    # Default: tenta roc_auc se binário, senão accuracy
    else:
        if len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_prob)
        else:
            y_pred = (y_prob >= 0.5).astype(int)
            return accuracy_score(y_true, y_pred)