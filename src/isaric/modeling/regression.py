"""
Regression models for the RAPID methodology.

This module provides functions to configure regression models and
concrete pipeline classes for logistic regression and generalized
linear models (GLM).

Functions (Configuration):
- create_logistic_model: Configure a logistic regression (binomial GLM).
- create_glm_model: Configure a generalized linear model.

Subclasses (Pipelines):
- LogisticRegression: Concrete pipeline for logistic regression.
- GLM: Concrete pipeline for generalized linear model.

Helper Functions:
- _prepare_data_from_vars: Prepare y/X from variable names.
- _prepare_data_from_formula: Prepare y/X from formula.
- _build_result_df: Build formatted results DataFrame.
- _map_variable_label: Apply display labels to variable names.
- _parse_variable_name: Parse variable names with interactions/categories.
- _validate_inputs: Validate user input parameters.
- _validate_binary_outcome: Validate that the outcome variable is binary.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def create_logistic_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    formula: Optional[str] = None,
    link: str = "logit"
) -> Tuple[sm.GLM, pd.DataFrame, pd.Series]:
    """
    Configure a logistic regression model (binomial GLM).

    This function prepares the data and returns a configured GLM
    object that is NOT yet fitted. The fit() method of the pipeline
    class will train the model.

    Args:
        data: Input DataFrame in ARC format.
        dependent_var: Name of the binary outcome variable (0/1).
        independent_vars: List of predictor variable names.
        formula: Patsy-style formula string (optional).
            If provided, overrides dependent_var and independent_vars.
        link: Link function for the binomial family.
            Options: "logit", "probit", "cloglog".

    Returns:
        Tuple of (model, X, y):
            - model: Configured statsmodels GLM (not fitted).
            - X: Predictor matrix (DataFrame).
            - y: Outcome vector (Series).

    Raises:
        ValueError: If outcome variable is not binary or columns are missing.
    """
    # Validate inputs
    _validate_inputs(data, dependent_var, independent_vars, formula)

    # Validate outcome is binary
    _validate_binary_outcome(data, dependent_var, formula)

    # Prepare data
    if formula:
        y, X = _prepare_data_from_formula(data, formula)
    else:
        y, X = _prepare_data_from_vars(data, dependent_var, independent_vars)

    # Configure link function
    link_map = {
        "logit": sm.families.links.Logit,
        "probit": sm.families.links.Probit,
        "cloglog": sm.families.links.CLogLog,
    }
    if link not in link_map:
        raise ValueError(
            f"Unknown link: {link}. Use 'logit', 'probit', or 'cloglog'."
        )

    # Configure GLM (binomial)
    model = sm.GLM(
        endog=y,
        exog=X,
        family=sm.families.Binomial(link=link_map[link]())
    )

    return model, X, y


def create_glm_model(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    formula: Optional[str] = None,
    family: str = "gaussian",
    link: str = "identity"
) -> Tuple[sm.GLM, pd.DataFrame, pd.Series]:
    """
    Configure a Generalized Linear Model (GLM).

    This function prepares the data and returns a configured GLM
    object that is NOT yet fitted. The fit() method of the pipeline
    class will train the model.

    Args:
        data: Input DataFrame in ARC format.
        dependent_var: Name of the outcome variable.
        independent_vars: List of predictor variable names.
        formula: Patsy-style formula string (optional).
            If provided, overrides dependent_var and independent_vars.
        family: Distribution family for the GLM.
            Options: "gaussian", "gamma", "inv_gaussian", "tweedie".
        link: Link function for the GLM.
            Options: "identity", "log", "inverse", "sqrt".

    Returns:
        Tuple of (model, X, y):
            - model: Configured statsmodels GLM (not fitted).
            - X: Predictor matrix (DataFrame).
            - y: Outcome vector (Series).

    Raises:
        ValueError: If family or link is unknown, or columns are missing.
    """
    # Validate inputs
    _validate_inputs(data, dependent_var, independent_vars, formula)

    # Prepare data
    if formula:
        y, X = _prepare_data_from_formula(data, formula)
    else:
        y, X = _prepare_data_from_vars(data, dependent_var, independent_vars)

    # Configure family
    family_map = {
        "gaussian": sm.families.Gaussian,
        "gamma": sm.families.Gamma,
        "inv_gaussian": sm.families.InverseGaussian,
        "tweedie": sm.families.Tweedie,
    }
    if family not in family_map:
        raise ValueError(
            f"Unknown family: {family}. "
            "Use 'gaussian', 'gamma', 'inv_gaussian', or 'tweedie'."
        )

    # Configure link
    link_map = {
        "identity": sm.families.links.Identity,
        "log": sm.families.links.Log,
        "inverse": sm.families.links.InversePower,
        "sqrt": sm.families.links.Sqrt,
    }
    if link not in link_map:
        raise ValueError(
            f"Unknown link: {link}. "
            "Use 'identity', 'log', 'inverse', or 'sqrt'."
        )

    # Configure GLM
    model = sm.GLM(
        endog=y,
        exog=X,
        family=family_map[family](link=link_map[link]())
    )

    return model, X, y


# ============================================================================
# PRIVATE HELPERS - DATA PREPARATION
# ============================================================================

def _prepare_data_from_vars(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str]
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare y and X from variable names.

    Args:
        data: Input DataFrame.
        dependent_var: Name of the outcome variable.
        independent_vars: List of predictor variable names.

    Returns:
        Tuple of (y, X).

    Raises:
        ValueError: If columns are not found.
    """
    for col in [dependent_var] + independent_vars:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    y = data[dependent_var]
    X = data[independent_vars]
    X = sm.add_constant(X)

    return y, X


def _prepare_data_from_formula(
    data: pd.DataFrame,
    formula: str
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare y and X from a Patsy-style formula.

    Args:
        data: Input DataFrame.
        formula: Patsy-style formula string (e.g., "outcome ~ age + sex").

    Returns:
        Tuple of (y, X).
    """
    import patsy

    y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    y = y.iloc[:, 0]

    return y, X


# ============================================================================
# PRIVATE HELPERS - RESULT DATAFRAME
# ============================================================================

def _build_result_df(
    fitted_model: sm.GLM,
    X: pd.DataFrame,
    y: pd.Series,
    labels: Optional[Dict[str, str]] = None,
    is_logistic: bool = False
) -> pd.DataFrame:
    """
    Build a formatted results DataFrame from the fitted model.

    For logistic regression, returns Odds Ratios with confidence
    intervals. For GLM, returns coefficients (or exponentiated
    coefficients if using a log link).

    Args:
        fitted_model: Fitted statsmodels GLM.
        X: Predictor matrix.
        y: Outcome vector.
        labels: Dictionary mapping variable names to display labels.
        is_logistic: If True, exponentiates coefficients to Odds Ratios.

    Returns:
        DataFrame with columns: Variable, Effect, LowerCI, UpperCI, p-value.
    """
    summary_table = fitted_model.summary2().tables[1]

    # Determine p-value column
    p_value_col = 'P>|z|' if 'P>|z|' in summary_table.columns else 'P>|t|'

    # Build result DataFrame
    result_df = summary_table[['Coef.', '[0.025', '0.975]', p_value_col]].reset_index()
    result_df = result_df.rename(columns={
        'index': 'Variable',
        'Coef.': 'Coefficient',
        '[0.025': 'LowerCI',
        '0.975]': 'UpperCI',
        p_value_col: 'p-value'
    })

    # Exponentiate for logistic or log-link
    if is_logistic:
        result_df[['Coefficient', 'LowerCI', 'UpperCI']] = np.exp(
            result_df[['Coefficient', 'LowerCI', 'UpperCI']]
        )
        result_df = result_df.rename(columns={'Coefficient': 'OddsRatio'})
    else:
        # Check if log link (for Gamma, Poisson, etc.)
        if isinstance(fitted_model.family.link, sm.families.links.Log):
            result_df[['Coefficient', 'LowerCI', 'UpperCI']] = np.exp(
                result_df[['Coefficient', 'LowerCI', 'UpperCI']]
            )
            result_df = result_df.rename(columns={'Coefficient': 'MeanRatio'})

    # Apply labels if provided
    if labels:
        result_df['Variable'] = result_df['Variable'].apply(
            lambda x: _parse_variable_name(x, labels)
        )

    # Filter intercept
    result_df = result_df[
        ~result_df['Variable'].str.lower().isin(['intercept', 'const', 'constant'])
    ]

    return result_df


def _map_variable_label(
    df: pd.DataFrame,
    labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Apply display labels to variable names in the DataFrame.

    Args:
        df: DataFrame with 'Variable' column.
        labels: Dictionary mapping raw names to display labels.

    Returns:
        DataFrame with labels applied.
    """
    if not labels:
        return df

    df = df.copy()
    df['Variable'] = df['Variable'].apply(
        lambda x: _parse_variable_name(x, labels)
    )
    return df


def _parse_variable_name(
    var_name: str,
    labels: Dict[str, str]
) -> str:
    """
    Parse a variable name and apply display labels.

    Handles variable names from Patsy formulas that may include
    interactions or categorical levels.

    Examples:
        "Intercept" → "Intercept"
        "C(sex)[T.Male]" → "Sex (Male)"
        "age" → "Age (years)"

    Args:
        var_name: Raw variable name.
        labels: Dictionary mapping base names to display labels.

    Returns:
        Formatted variable name.
    """
    if var_name == 'Intercept':
        return labels.get('Intercept', 'Intercept')
    elif '[' in var_name:
        # Categorical variable with level: C(sex)[T.Male]
        base_var = var_name.split('[')[0].replace('C(', '').replace(')', '').strip()
        level = var_name.split('[')[1].split(']')[0]
        base_label = labels.get(base_var, base_var)
        return f'{base_label} ({level})'
    else:
        # Simple variable name
        var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
        return labels.get(var_name_clean, var_name_clean)


# ============================================================================
# PRIVATE HELPERS - VALIDATION
# ============================================================================

def _validate_inputs(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: List[str],
    formula: Optional[str] = None
) -> None:
    """
    Validate user input parameters.

    Args:
        data: Input DataFrame.
        dependent_var: Name of the outcome variable.
        independent_vars: List of predictor variable names.
        formula: Patsy-style formula string (optional).

    Raises:
        ValueError: If any input is invalid.
    """
    if data is None:
        raise ValueError("data cannot be None")

    if data.empty:
        raise ValueError("data cannot be empty")

    if formula is None:
        if not dependent_var:
            raise ValueError(
                "dependent_var cannot be None or empty if formula is not provided."
            )

        if not independent_vars:
            raise ValueError(
                "independent_vars cannot be None or empty if formula is not provided."
            )

    if dependent_var and dependent_var not in data.columns:
        raise ValueError(
            f"Outcome variable '{dependent_var}' not found in data columns."
        )

    if independent_vars:
        missing_vars = [p for p in independent_vars if p not in data.columns]
        if missing_vars:
            raise ValueError(
                f"Predictor(s) not found in data columns: {missing_vars}"
            )


def _validate_binary_outcome(
    data: pd.DataFrame,
    dependent_var: str,
    formula: Optional[str] = None
) -> None:
    """
    Validate that the outcome variable is binary and coded as 0/1.

    Args:
        data: Input DataFrame.
        dependent_var: Name of the outcome variable.
        formula: Patsy-style formula string (optional).

    Raises:
        ValueError: If outcome is not binary or not coded as 0/1.
    """
    if dependent_var:
        outcome = data[dependent_var].dropna()
    elif formula:
        outcome_name = formula.split('~')[0].strip()
        outcome = data[outcome_name].dropna()
    else:
        return

    unique_values = outcome.unique()

    if len(unique_values) != 2:
        raise ValueError(
            f"Logistic regression requires a binary outcome variable. "
            f"Found {len(unique_values)} unique values: {unique_values}"
        )

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Outcome variable must be coded as 0 and 1. "
            f"Found values: {sorted(unique_values)}"
        )


# ============================================================================
# SUBCLASSES (INHERIT FROM RAPID)
# ============================================================================

from isaric.rapid import RAPID


class LogisticRegression(RAPID):
    """
    Concrete pipeline for Logistic Regression.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: sm.GLM,
        X: pd.DataFrame,
        y: pd.Series,
        dependent_var: str,
        independent_vars: List[str],
        labels: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize LogisticRegression with configured model and data.

        Args:
            model: Configured statsmodels GLM (from create_logistic_model).
            X: Predictor matrix.
            y: Outcome vector.
            dependent_var: Outcome variable name.
            independent_vars: Predictor variable names.
            labels: Dictionary for variable display labels.
        """
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.labels = labels
        self.model_type = "logistic"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        # Define plots_map with available plots
        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for Logistic Regression."""
        from isaric.visualization.forestplots import odds_ratio_plot
        from isaric.visualization.heatmaps import confusion_matrix_heatmap

        self.plots_map = {
            "forest_plot": self._forest_plot,
            "confusion_matrix": self._confusion_matrix,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "logistic",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        link: str = "logit",
        labels: Optional[Dict[str, str]] = None,
        **params
    ) -> "LogisticRegression":
        """
        Configure and instantiate the Logistic Regression pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "logistic").
            dependent_var: Binary outcome variable (0/1).
            independent_vars: Predictor variable names.
            formula: Patsy-style formula (optional).
            link: Link function ("logit", "probit", "cloglog").
            labels: Dictionary for variable display labels.

        Returns:
            LogisticRegression instance ready for training.
        """
        model_config, X, y = create_logistic_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            formula=formula,
            link=link
        )

        return cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            labels=labels,
            **params
        )

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _forest_plot(self):
        """Generate forest plot for Odds Ratios."""
        from isaric.visualization.forestplots import odds_ratio_plot
        fig = odds_ratio_plot(
            self.result_df,
            effect_col='OddsRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Odds Ratios (Logistic Regression)"
        )
        return fig

    def _confusion_matrix(self):
        """Generate confusion matrix heatmap."""
        from isaric.visualization.heatmaps import confusion_matrix_heatmap

        y_prob = self.fitted_model.fittedvalues
        y_pred = (y_prob >= 0.5).astype(int)

        fig = confusion_matrix_heatmap(
            y_true=self.y,
            y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Logistic Regression"
        )
        return fig


class GLM(RAPID):
    """
    Concrete pipeline for Generalized Linear Model (GLM).

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: sm.GLM,
        X: pd.DataFrame,
        y: pd.Series,
        dependent_var: str,
        independent_vars: List[str],
        family: str = "gaussian",
        link: str = "identity",
        labels: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize GLM with configured model and data.

        Args:
            model: Configured statsmodels GLM (from create_glm_model).
            X: Predictor matrix.
            y: Outcome vector.
            dependent_var: Outcome variable name.
            independent_vars: Predictor variable names.
            family: Distribution family.
            link: Link function.
            labels: Dictionary for variable display labels.
        """
        self._model = model
        self.X = X
        self.y = y
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.family = family
        self.link = link
        self.labels = labels
        self.model_type = "glm"
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for GLM."""
        self.plots_map = {
            "forest_plot": self._forest_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "glm",
        dependent_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        family: str = "gaussian",
        link: str = "identity",
        labels: Optional[Dict[str, str]] = None,
        **params
    ) -> "GLM":
        """
        Configure and instantiate the GLM pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "glm").
            dependent_var: Outcome variable.
            independent_vars: Predictor variable names.
            formula: Patsy-style formula (optional).
            family: Distribution family.
            link: Link function.
            labels: Dictionary for variable display labels.

        Returns:
            GLM instance ready for training.
        """
        model_config, X, y = create_glm_model(
            data=data,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            formula=formula,
            family=family,
            link=link
        )

        return cls(
            model=model_config,
            X=X,
            y=y,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            family=family,
            link=link,
            labels=labels,
            **params
        )

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _forest_plot(self):
        """Generate forest plot for coefficients."""
        from isaric.visualization.forestplots import coefficient_plot

        fig = coefficient_plot(
            self.result_df,
            effect_col='Coefficient',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Coefficients (GLM)"
        )
        return fig