"""
Descriptive statistics for the RAPID methodology.

This module provides functions to configure and calculate descriptive
statistics for the RAPID analytical pipeline. Descriptive statistics
provide simple quantitative descriptions of the data.

Functions (Configuration):
- create_descriptive_model: Configure descriptive analysis.

Subclasses (Pipelines):
- Descriptive: Concrete pipeline for descriptive statistics.

Helper Functions:
- _build_result_df: Build consolidated descriptive statistics DataFrame.
- _calculate_frequency_table: Calculate frequency for categorical variables.
- _calculate_continuous_table: Calculate summary for continuous variables.
- _is_categorical: Determine if a series is categorical.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from isaric.rapid import RAPID


# ============================================================================
# PUBLIC FUNCTIONS (CONFIGURATION)
# ============================================================================

def create_descriptive_model(
    data: pd.DataFrame,
    variables: List[str]
) -> Tuple[None, pd.DataFrame]:
    """
    Configure descriptive analysis (not executed).

    Args:
        data: Input DataFrame in ARC format.
        variables: Variable names to include.

    Returns:
        Tuple of (None, data_selected).

    Raises:
        ValueError: If variables are not found.
    """
    for col in variables:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    data_selected = data[variables].copy()
    return None, data_selected


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _build_result_df(
    data: pd.DataFrame,
    variables: List[str]
) -> pd.DataFrame:
    """
    Build consolidated descriptive statistics DataFrame.

    Args:
        data: DataFrame with selected variables.
        variables: Variable names.

    Returns:
        DataFrame with descriptive statistics.
    """
    results = []

    for col in variables:
        if _is_categorical(data[col]):
            result = _calculate_frequency_table(data, col)
        else:
            result = _calculate_continuous_table(data, col)

        result['Variable'] = col
        results.append(result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def _calculate_frequency_table(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Calculate frequency table for categorical variable.

    Args:
        data: DataFrame containing the column.
        column: Categorical variable name.

    Returns:
        DataFrame with Category, Count, Percentage.
    """
    value_counts = data[column].value_counts(dropna=False)
    total = value_counts.sum()

    freq_df = pd.DataFrame({
        'Category': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / total * 100).round(2)
    })

    return freq_df


def _calculate_continuous_table(
    data: pd.DataFrame,
    column: str,
    method: str = "auto"
) -> pd.DataFrame:
    """
    Calculate summary statistics for continuous variable.

    Args:
        data: DataFrame containing the column.
        column: Continuous variable name.
        method: "auto", "mean_sd", or "median_iqr".

    Returns:
        DataFrame with descriptive statistics.

    Raises:
        ValueError: If method is invalid.
    """
    if method not in ("auto", "mean_sd", "median_iqr"):
        raise ValueError(
            f"method must be 'auto', 'mean_sd', or 'median_iqr'. "
            f"Received: {method}"
        )

    series = data[column].dropna()

    if method == "auto":
        skewness = series.skew()
        method = "median_iqr" if abs(skewness) > 1 else "mean_sd"

    if method == "mean_sd":
        result = pd.DataFrame({
            'Mean': [round(series.mean(), 2)],
            'Std_Dev': [round(series.std(), 2)],
            'Min': [series.min()],
            'Max': [series.max()]
        })

    elif method == "median_iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        result = pd.DataFrame({
            'Median': [round(series.median(), 2)],
            'Q1': [round(q1, 2)],
            'Q3': [round(q3, 2)],
            'IQR': [round(q3 - q1, 2)]
        })

    return result


def _is_categorical(series: pd.Series) -> bool:
    """
    Determine if a series is categorical.

    Args:
        series: Input series.

    Returns:
        True if categorical, False otherwise.
    """
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return True

    n_unique = series.nunique()
    return n_unique <= 10


# ============================================================================
# SUBCLASS (INHERIT FROM RAPID)
# ============================================================================

class Descriptive(RAPID):
    """
    Concrete pipeline for descriptive statistics.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: None,
        X: pd.DataFrame,
        variables: List[str],
        **kwargs
    ):
        """
        Initialize Descriptive with data.

        Args:
            model: None (descriptive is model-free).
            X: DataFrame with selected variables.
            variables: Variable names.
        """
        self._model = model
        self.X = X
        self.variables = variables
        self.model_type = "descriptive"
        self.y = None
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for Descriptive."""
        self.plots_map = {
            "bar_plot": self._bar_plot,
            "correlation_heatmap": self._correlation_heatmap,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "descriptive",
        variables: Optional[List[str]] = None,
        **params
    ) -> "Descriptive":
        """
        Configure and instantiate the Descriptive pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "descriptive").
            variables: Variable names to analyze.

        Returns:
            Descriptive instance ready for analysis.
        """
        model_config, X = create_descriptive_model(
            data=data,
            variables=variables
        )

        return cls(
            model=model_config,
            X=X,
            variables=variables,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self):
        """Not applicable for descriptive."""
        return None

    def _build_result_df(self):
        """Build descriptive statistics DataFrame."""
        return _build_result_df(
            data=self.X,
            variables=self.variables
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate basic descriptive metrics."""
        metrics = {}
        for col in self.variables:
            if _is_categorical(self.X[col]):
                metrics[col] = {
                    'n_unique': int(self.X[col].nunique()),
                    'most_common': str(self.X[col].mode()[0]) if not self.X[col].mode().empty else None,
                }
            else:
                metrics[col] = {
                    'mean': float(self.X[col].mean()),
                    'std': float(self.X[col].std()),
                    'median': float(self.X[col].median()),
                }
        return metrics

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Not applicable for descriptive."""
        return None

    def _calibration_curve(self):
        """Not applicable for descriptive."""
        return None

    def _check_assumptions(self):
        """Not applicable for descriptive."""
        return None

    def _train_test_split(self, test_size=0.2):
        """Not applicable for descriptive."""
        return None

    def _validate_external(self, external_data):
        """Not applicable for descriptive."""
        return None

    def _validate_bootstrap(self, n_iterations=1000):
        """Bootstrap validation for descriptive statistics."""
        return None

    def _validate_sensitivity(self):
        """Not applicable for descriptive."""
        return None

    def _validate_subgroups(self, subgroups):
        """Not applicable for descriptive."""
        return None

    def _validate_net_benefit(self):
        """Not applicable for descriptive."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _bar_plot(self, backend="plotly"):
        """Generate bar plot for first categorical variable."""
        from isaric.visualization.barplots import simple_bar_plot

        categorical_vars = [v for v in self.variables if _is_categorical(self.X[v])]

        if not categorical_vars:
            return None

        first_cat = categorical_vars[0]
        value_counts = self.X[first_cat].value_counts().reset_index()
        value_counts.columns = [first_cat, 'Count']

        return simple_bar_plot(
            value_counts,
            x_col=first_cat,
            y_col='Count',
            title=f"Distribution of {first_cat}",
            backend=backend
        )

    def _correlation_heatmap(self, backend="plotly"):
        """Generate correlation heatmap for numeric variables."""
        from isaric.visualization.heatmaps import correlation_heatmap

        numeric_vars = [
            v for v in self.variables
            if pd.api.types.is_numeric_dtype(self.X[v])
        ]

        if len(numeric_vars) < 2:
            return None

        return correlation_heatmap(
            self.X[numeric_vars],
            title="Correlation Heatmap - Descriptive",
            backend=backend
        )