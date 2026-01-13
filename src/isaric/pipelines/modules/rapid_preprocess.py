"""
Modular preprocessing utilities for statistical modeling.
Uses Formulaic to handle model matrices, encoding, and transformations.
"""

import pandas as pd
import numpy as np
from formulaic import Formula


class FormulaProcessor:
    """
    Handles data transformation using formulas.
    Encapsulates encoding, numerical transformations, and matrix generation.
    """

    @staticmethod
    def build_formula(target_cols, predictor_cols, intercept=True):
        """
        Constructs an R-style formula string from lists of column names.
        
        Args:
            target_cols: List of column names for the dependent variables.
            predictor_cols: List of column names for the predictors.
            intercept: Boolean to include or exclude the intercept (+1 or -1).
            
        Returns:
            A formula string (e.g., 'time + event ~ age + sex + 1').
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
            y, X: Target and Predictor matrices as Pandas DataFrames.
        """
        # Single line logic using Formulaic to handle encoding and model matrix generation
        y, X = Formula(formula).get_model_matrix(df)
        
        if ensure_numeric:
            y = y.apply(pd.to_numeric, errors='coerce')
            X = X.apply(pd.to_numeric, errors='coerce')
            
        return y, X


class RapidPreprocessor:
    """
    Unified interface for preprocessing.
    Provides convenience access to formula-based transformations.
    """
    
    formula = FormulaProcessor
    
    @staticmethod
    def prepare_data(df, target_cols, predictor_cols, intercept=True):
        """
        High-level helper to prepare data for statistical models.
        Automatically builds the formula and generates matrices.
        
        Args:
            df: Input DataFrame.
            target_cols: List of target column names.
            predictor_cols: List of predictor column names.
            intercept: Whether to include an intercept term.
            
        Returns:
            y: Target matrix.
            X: Feature matrix (encoded and transformed).
            predictors: List of resulting column names in X.
        """
        # Build formula internally
        formula_str = FormulaProcessor.build_formula(target_cols, predictor_cols, intercept)
        
        # Get matrices using the generated formula
        y, X = FormulaProcessor.get_matrices(df, formula_str)
        
        return y, X, X.columns.tolist()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate usage of the modular preprocessing utilities."""
    
    data = pd.DataFrame({
        'time': [10, 12, 15, 20],
        'event': [1, 0, 1, 1],
        'age': [25, 30, 35, 40],
        'group': ['A', 'B', 'A', 'C'],
        'bmi': [22.5, 27.0, 24.1, 29.0]
    })
    
    # User only provides lists of columns, the module handles the formula
    y_mat, X_mat, predictors = RapidPreprocessor.prepare_data(
        df=data,
        target_cols=['time', 'event'],
        predictor_cols=['age', 'group', 'bmi'],
        intercept=True
    )
    
    print("Processed Predictors:", predictors)
    print("\nFeature Matrix (X):")
    print(X_mat.head())


if __name__ == "__main__":
    # example_usage()
    pass