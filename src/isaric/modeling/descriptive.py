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
    variables: List[str],
    group_col: Optional[str] = None
) -> Tuple[None, pd.DataFrame]:
    """
    Configure descriptive analysis (not executed).

    Args:
        data: Input DataFrame in ARC format.
        variables: Variable names to include.
        group_col: Optional column for grouping (e.g., treatment arm).

    Returns:
        Tuple of (None, data_selected).

    Raises:
        ValueError: If variables or group_col are not found.
    """
    for col in variables:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if group_col is not None:
        if group_col not in data.columns:
            raise ValueError(f"Column '{group_col}' not found in DataFrame.")
        if group_col not in variables:
            variables = list(variables) + [group_col]

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

    Order of precedence:
    1. Ordinal (CategoricalDtype with ordered=True) → frequency + tendency
    2. Categorical (CategoricalDtype, object, bool, binary) → frequency
    3. Continuous (numeric) → mean/std/median/etc.

    Args:
        data: DataFrame with selected variables.
        variables: Variable names.

    Returns:
        DataFrame with descriptive statistics.
    """
    results = []

    for col in variables:
        if _is_ordinal(data[col]):
            result = _calculate_ordinal_table(data, col)
        elif _is_categorical(data[col]):
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

    In addition to Category/Count/Percentage, the mode (most frequent
    value) is added as an extra row.

    Args:
        data: DataFrame containing the column.
        column: Categorical variable name.

    Returns:
        DataFrame with Category, Count, Percentage and an extra
        "Mode" row.
    """
    value_counts = data[column].value_counts(dropna=False)
    total = value_counts.sum()

    rows = []
    for cat, count in value_counts.items():
        rows.append({
            'Category': cat,
            'Count': int(count),
            'Percentage': round(count / total * 100, 2),
        })

    # Mode (most frequent value)
    mode_values = data[column].mode(dropna=True)
    if not mode_values.empty:
        mode_val = mode_values.iloc[0]
        mode_count = int((data[column] == mode_val).sum())
        rows.append({
            'Category': f'Mode ({mode_val})',
            'Count': mode_count,
            'Percentage': round(mode_count / total * 100, 2),
        })

    return pd.DataFrame(rows)


def _calculate_continuous_table(
    data: pd.DataFrame,
    column: str,
    method: str = "auto"
) -> pd.DataFrame:
    """
    Calculate summary statistics for continuous variable.

    Reports central tendency, dispersion, shape, and normality test.
    Output format: one row per variable, one column per statistic.
    """
    from scipy import stats

    if method not in ("auto", "mean_sd", "median_iqr"):
        raise ValueError(
            f"method must be 'auto', 'mean_sd', or 'median_iqr'. "
            f"Received: {method}"
        )

    series = data[column].dropna()

    if len(series) == 0:
        return pd.DataFrame()

    if method == "auto":
        skewness = series.skew()
        method = "median_iqr" if abs(skewness) > 1 else "mean_sd"

    # Central tendency + dispersion
    if method == "mean_sd":
        row = {
            'Mean': round(series.mean(), 4),
            'Std_Dev': round(series.std(), 4),
        }
    elif method == "median_iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        row = {
            'Median': round(series.median(), 4),
            'Q1': round(q1, 4),
            'Q3': round(q3, 4),
            'IQR': round(q3 - q1, 4),
        }

    # Range
    row['Min'] = round(series.min(), 4)
    row['Max'] = round(series.max(), 4)

    # Shape
    if len(series) >= 3:
        row['Skewness'] = round(series.skew(), 4)
    if len(series) >= 4:
        row['Kurtosis'] = round(series.kurt(), 4)

    # Normality (Shapiro-Wilk)
    try:
        if len(series) > 5000:
            import numpy as np
            rng = np.random.default_rng(42)
            sample = rng.choice(series.values, 5000, replace=False)
        else:
            sample = series.values

        if len(sample) >= 3:
            w_stat, p_value = stats.shapiro(sample)
            row['Shapiro_W'] = round(float(w_stat), 4)
            row['Shapiro_p'] = round(float(p_value), 4)
    except Exception:
        pass

    return pd.DataFrame([row])


def _is_categorical(series: pd.Series) -> bool:
    """
    Determine if a series is categorical.

    Follows the pandas dtype and preserves legacy behavior for binary
    numeric columns:

    - CategoricalDtype       → True (user-declared)
    - object (strings)       → True
    - bool                   → True
    - numeric with 2 unique values in {0, 1} → True (binary 0/1)
    - other numeric          → False (never inferred as categorical)

    The user is expected to cast the column to `category` dtype if it
    should be treated as categorical when the numeric heuristic does
    not apply (e.g., ICD codes, region names with many levels).

    Args:
        series: Input series.

    Returns:
        True if categorical, False otherwise.
    """
    # 1. User-declared categorical
    if isinstance(series.dtype, pd.CategoricalDtype):
        return True

    # 2. Strings
    if pd.api.types.is_object_dtype(series):
        return True

    # 3. Booleans
    if pd.api.types.is_bool_dtype(series):
        return True

    # 4. Binary 0/1 (legacy behavior for clinical variables like sex, smoking)
    if pd.api.types.is_numeric_dtype(series):
        unique = series.dropna().unique()
        if len(unique) == 2 and set(unique).issubset({0, 1, 0.0, 1.0}):
            return True

    # 5. Other numeric → continuous
    return False

def _is_ordinal(series: pd.Series) -> bool:
    """
    Determine if a series is ordinal.

    A series is ordinal if it has CategoricalDtype with `ordered=True`.
    Ordinal variables are treated differently from nominal categoricals:
    both frequency and central tendency (median, IQR) are reported.

    Args:
        series: Input series.

    Returns:
        True if ordinal, False otherwise.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        return bool(series.dtype.ordered)
    return False


def _calculate_ordinal_table(
    data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Calculate frequency + central tendency for ordinal variable.

    Args:
        data: DataFrame containing the column.
        column: Ordinal variable name.

    Returns:
        DataFrame with Category, Count, Percentage, and (as extra rows)
        Median, IQR, Mode.
    """
    series = data[column].dropna()

    # Frequência por categoria (mantendo a ordem)
    value_counts = series.value_counts(dropna=False)
    # Reordenar pelas categorias (respeita a ordem definida)
    if hasattr(series.dtype, 'categories'):
        ordered_cats = [
            c for c in series.dtype.categories if c in value_counts.index
        ]
        # Adiciona categorias que possam ter aparecido como NaN
        remaining = [c for c in value_counts.index if c not in ordered_cats]
        value_counts = value_counts.reindex(ordered_cats + remaining)

    total = value_counts.sum()

    rows = []
    for cat, count in value_counts.items():
        rows.append({
            'Category': cat,
            'Count': int(count),
            'Percentage': round(count / total * 100, 2),
        })

    # Estatísticas de tendência central
    # Para ordinal, median e IQR fazem sentido (assumindo escala numérica)
    # Se as categorias são strings não-numéricas, só reportamos moda
    try:
        # Tenta codificar as categorias como inteiros (1..N) na ordem definida
        if hasattr(series.dtype, 'categories'):
            mapping = {c: i+1 for i, c in enumerate(series.dtype.categories)}
            numeric = series.map(mapping)
            median_val = numeric.median()
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr_val = q3 - q1
            mode_val = series.mode().iloc[0] if not series.mode().empty else None
            rows.append({
                'Category': 'Median (rank)',
                'Count': '',
                'Percentage': round(median_val, 2),
            })
            rows.append({
                'Category': 'IQR (rank)',
                'Count': '',
                'Percentage': round(iqr_val, 2),
            })
            rows.append({
                'Category': 'Mode',
                'Count': '',
                'Percentage': mode_val,
            })
    except Exception:
        pass

    return pd.DataFrame(rows)


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
        group_col: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Descriptive with data.

        Args:
            model: None (descriptive is model-free).
            X: DataFrame with selected variables.
            variables: Variable names.
            group_col: Optional column for grouping (e.g., treatment arm).
        """
        self._model = model
        self.X = X
        self.variables = variables
        self.group_col = group_col
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
            "pairplot": self._pairplot,
            "histograms": self._histograms,
            "boxplots": self._boxplots,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "descriptive",
        variables: Optional[List[str]] = None,
        group_col: Optional[str] = None,
        **params
    ) -> "Descriptive":
        """
        Configure and instantiate the Descriptive pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "descriptive").
            variables: Variable names to analyze.
            group_col: Optional column for grouping (e.g., treatment arm).

        Returns:
            Descriptive instance ready for analysis.
        """
        model_config, X = create_descriptive_model(
            data=data,
            variables=variables,
            group_col=group_col
        )

        instance = cls(
            model=model_config,
            X=X,
            variables=variables,
            group_col=group_col,
            **params
        )
        instance._random_seed = 42

        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """Not applicable for descriptive."""
        return None

    def _build_result_df(self):
        """Build descriptive statistics DataFrame."""
        return _build_result_df(
            data=self.X,
            variables=self.variables
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate descriptive metrics."""
        result = {}
        for col in self.variables:
            if _is_ordinal(self.X[col]):
                series = self.X[col].dropna()
                try:
                    if hasattr(series.dtype, 'categories'):
                        mapping = {c: i+1 for i, c in enumerate(series.dtype.categories)}
                        numeric = series.map(mapping)
                        median_rank = float(numeric.median())
                        iqr_rank = float(numeric.quantile(0.75) - numeric.quantile(0.25))
                    else:
                        median_rank = None
                        iqr_rank = None
                except Exception:
                    median_rank = None
                    iqr_rank = None

                result[col] = {
                    'type': 'ordinal',
                    'n_unique': int(series.nunique()),
                    'mode': str(series.mode().iloc[0]) if not series.mode().empty else None,
                    'median_rank': median_rank,
                    'iqr_rank': iqr_rank,
                }
            elif _is_categorical(self.X[col]):
                result[col] = {
                    'type': 'categorical',
                    'n_unique': int(self.X[col].nunique()),
                    'most_common': str(self.X[col].mode()[0]) if not self.X[col].mode().empty else None,
                }
            else:
                result[col] = {
                    'type': 'continuous',
                    'mean': float(self.X[col].mean()),
                    'std': float(self.X[col].std()),
                    'median': float(self.X[col].median()),
                }

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "pandas",
            "method": "Descriptive Statistics",
            "random_seed": self._random_seed,
            "N_observations": len(self.X),
            "N_variables": len(self.variables),
            "parameters": {
                "variables": self.variables,
                "group_col": self.group_col,
            },
        }

        # Se group_col: adiciona tamanho de cada grupo
        if self.group_col is not None and self.group_col in self.X.columns:
            grouped_counts = (
                self.X[self.group_col]
                .value_counts(dropna=False)
                .to_dict()
            )
            info["group_sizes"] = {
                str(k): int(v) for k, v in grouped_counts.items()
            }

        self.model_info = info

        return result

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
        from isaric.visualization.barplots import (
            simple_bar_plot,
            grouped_bar_plot,
        )

        categorical_vars = [
            v for v in self.variables
            if _is_categorical(self.X[v]) and v != self.group_col
        ]

        if not categorical_vars:
            return None

        first_cat = categorical_vars[0]

        if self.group_col is not None:
            # Agrupado por group_col
            # Prepara dados: contagem por (categoria, grupo)
            # Como barplots espera colunas, criamos uma coluna sintética
            # de contagem por grupo
            df = self.X[[first_cat, self.group_col]].dropna()
            # Conta cada combinação
            counts = (
                df.groupby([first_cat, self.group_col])
                .size()
                .reset_index(name='Count')
            )
            return grouped_bar_plot(
                counts,
                x_col=first_cat,
                y_col='Count',
                group_col=self.group_col,
                title=f"Distribution of {first_cat} by {self.group_col}",
                backend=backend
            )
        else:
            # Simples
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

    def _pairplot(self, backend="plotly"):
        """Generate pairplot for numeric variables."""
        from isaric.visualization.heatmaps import pairplot

        numeric_vars = [
            v for v in self.variables
            if pd.api.types.is_numeric_dtype(self.X[v])
        ]

        if len(numeric_vars) < 2:
            raise ValueError(
                "Pairplot requires at least 2 numeric variables. "
                f"Found: {len(numeric_vars)}"
            )

        # group_col é opcional; será passado em E.7
        group_col = getattr(self, 'group_col', None)

        return pairplot(
            self.X,
            variables=numeric_vars,
            group_col=group_col,
            title="Pairplot - Descriptive",
            backend=backend
        )

    def _histograms(self, backend="plotly"):
        """Generate histograms for each continuous variable."""
        from isaric.visualization.barplots import histogram_plot

        numeric_vars = [
            v for v in self.variables
            if pd.api.types.is_numeric_dtype(self.X[v]) and v != self.group_col
        ]

        if not numeric_vars:
            raise ValueError("No continuous variables found for histograms.")

        figs = []
        for var in numeric_vars:
            fig = histogram_plot(
                self.X,
                column=var,
                group_col=self.group_col,
                title=f"Distribution of {var}",
                backend=backend
            )
            figs.append(fig)

        return figs

    def _boxplots(self, backend="plotly"):
        """Generate boxplots for each continuous variable."""
        from isaric.visualization.barplots import boxplot

        numeric_vars = [
            v for v in self.variables
            if pd.api.types.is_numeric_dtype(self.X[v]) and v != self.group_col
        ]

        if not numeric_vars:
            raise ValueError("No continuous variables found for boxplots.")

        figs = []
        for var in numeric_vars:
            fig = boxplot(
                self.X,
                column=var,
                group_col=self.group_col,
                title=f"Boxplot of {var}",
                backend=backend
            )
            figs.append(fig)

        return figs