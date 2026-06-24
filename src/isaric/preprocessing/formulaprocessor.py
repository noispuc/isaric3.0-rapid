"""
Preprocessing utilities for formula-based model matrix construction.
Uses Formulaic to handle R-style formulas, encoding, and transformations.
"""

import pandas as pd
import numpy as np
from formulaic import Formula


class FormulaProcessor:
    """
    Handles data transformation using R-style formulas.
    Encapsulates encoding, numerical transformations, and matrix generation.
    """

    @staticmethod
    def build_formula(target_cols, predictor_cols, intercept=True):
        """
        Constructs an R-style formula string from lists of column names.

        Args:
            target_cols: List of column names for the dependent variables.
            predictor_cols: List of column names for the predictors.
            intercept: Whether to include an intercept term (+1 or -1).

        Returns:
            A formula string (e.g., 'y ~ x1 + x2 + 1').
        """
        y_str = " + ".join(target_cols)
        x_str = " + ".join(predictor_cols)
        suffix = " + 1" if intercept else " - 1"
        return f"{y_str} ~ {x_str}{suffix}"

    @staticmethod
    def get_matrices(df, formula, ensure_numeric=True):
        """
        Converts a DataFrame and formula string into y (target) and X (predictors) matrices.

        Args:
            df: Input DataFrame.
            formula: R-style formula string.
            ensure_numeric: Whether to force numeric conversion on the resulting matrices.

        Returns:
            y, X: Target and predictor matrices as pandas DataFrames.
        """
        y, X = Formula(formula).get_model_matrix(df)

        if ensure_numeric:
            y = y.apply(pd.to_numeric, errors='coerce')
            X = X.apply(pd.to_numeric, errors='coerce')

        return y, X


class RapidPreprocessor:
    """
    Unified interface for formula-based preprocessing.
    Convenience wrapper around FormulaProcessor.
    """

    formula = FormulaProcessor

    @staticmethod
    def prepare_data(df, formula=None, target_cols=None, predictor_cols=None, intercept=True):
        """
        High-level helper to prepare data matrices for statistical models.

        Args:
            df: Input DataFrame.
            formula: Optional. Full R-style formula string. Overrides col lists if provided.
            target_cols: Optional. List of target column names.
            predictor_cols: Optional. List of predictor column names.
            intercept: Whether to include an intercept term (only used if formula is None).

        Returns:
            y: Target matrix.
            X: Feature matrix (encoded and transformed).
            predictors: List of resulting column names in X.
        """
        if formula is None:
            if target_cols is None or predictor_cols is None:
                raise ValueError(
                    "Either 'formula' or both 'target_cols' and 'predictor_cols' must be provided."
                )
            formula = FormulaProcessor.build_formula(target_cols, predictor_cols, intercept)

        y, X = FormulaProcessor.get_matrices(df, formula)
        return y, X, X.columns.tolist()
