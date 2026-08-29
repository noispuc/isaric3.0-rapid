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

    cv = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

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

    cv = RepeatedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

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

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

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