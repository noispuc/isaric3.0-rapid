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
    def prepare_data(df, formula=None, target_cols=None, predictor_cols=None, intercept=True):
        """
        High-level helper to prepare data for statistical models.
        
        Args:
            df: Input DataFrame.
            formula: Optional. Full R-style formula string. If provided, overrides col lists.
            target_cols: Optional. List of target column names.
            predictor_cols: Optional. List of predictor column names.
            intercept: Whether to include an intercept term (only used if formula is None).
            
        Returns:
            y: Target matrix.
            X: Feature matrix (encoded and transformed).
            predictors: List of resulting column names in X.
        """
        # Logic to decide between provided formula or building one from lists
        if formula is None:
            if target_cols is None or predictor_cols is None:
                raise ValueError("Either 'formula' or both 'target_cols' and 'predictor_cols' must be provided.")
            
            formula = FormulaProcessor.build_formula(target_cols, predictor_cols, intercept)
        
        y, X = FormulaProcessor.get_matrices(df, formula)
        
        return y, X, X.columns.tolist()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def test_preprocessing():
    """Demonstrate the different ways to use the preprocessor."""
    
    data = pd.DataFrame({
        'time': [10, 12, 15, 20],
        'event': [1, 0, 1, 1],
        'age': [25, 30, 35, 40],
        'treatment': ['Placebo', 'Drug', 'Placebo', 'Drug'],
        'sex': ['M', 'F', 'M', 'F']
    })

    # OPTION 1: Simple lists (Convenience)
    print("--- Option 1: Using lists ---")
    y1, X1, cols1 = RapidPreprocessor.prepare_data(
        df=data, 
        target_cols=['time', 'event'], 
        predictor_cols=['age', 'sex']
    )
    print(f"Columns: {cols1}")

    # OPTION 2: Full Formula (Power & Control)
    # We specify 'Placebo' as the reference and add an interaction between age and treatment
    print("\n--- Option 2: Using complex formula ---")
    complex_formula = "time + event ~ age * C(treatment, Treatment(reference='Placebo')) + sex"
    
    y2, X2, cols2 = RapidPreprocessor.prepare_data(
        df=data, 
        formula=complex_formula
    )
    print(f"Columns: {cols2}")
    print(X2.head())


if __name__ == "__main__":
    # test_preprocessing()
    pass