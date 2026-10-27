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
- _prepare_data_from_formula: Prepare y/X from formula (uses formulaic).
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
    Prepare y and X from a formula string.

    Args:
        data: Input DataFrame.
        formula: Formula string (e.g., "outcome ~ age + sex").

    Returns:
        Tuple of (y, X).
    """
    from formulaic import model_matrix

    y, X = model_matrix(formula, data)
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

    Always includes raw Coefficient (log-OR for logistic), std_error,
    z_value, analytical LowerCI, UpperCI and p-value.

    For logistic regression also includes OddsRatio (exp(Coefficient)).
    For GLM with log-link also includes MeanRatio (exp(Coefficient)).

    full format: all columns.
    short format (controlled by summary()): Variable, main effect, p-value.

    Args:
        fitted_model: Fitted statsmodels GLM.
        X: Predictor matrix.
        y: Outcome vector.
        labels: Dictionary mapping variable names to display labels.
        is_logistic: If True, adds OddsRatio column.

    Returns:
        DataFrame with columns: Variable, Coefficient, [OddsRatio],
        std_error, z_value, LowerCI, UpperCI, p-value.
    """
    summary_table = fitted_model.summary2().tables[1]

    # Determine column names (vary slightly across statsmodels versions)
    p_value_col = 'P>|z|' if 'P>|z|' in summary_table.columns else 'P>|t|'
    z_col       = 'z'     if 'z'     in summary_table.columns else 't'

    result_df = summary_table[
        ['Coef.', 'Std.Err.', z_col, '[0.025', '0.975]', p_value_col]
    ].reset_index()

    result_df = result_df.rename(columns={
        'index':     'Variable',
        'Coef.':     'Coefficient',
        'Std.Err.':  'std_error',
        z_col:       'z_value',
        '[0.025':    'LowerCI',
        '0.975]':    'UpperCI',
        p_value_col: 'p-value',
    })

    # Logistic: add OddsRatio; CIs on OR scale; keep raw Coefficient (log-OR)
    if is_logistic:
        result_df['OddsRatio'] = np.exp(result_df['Coefficient'])
        result_df['LowerCI']   = np.exp(result_df['LowerCI'])
        result_df['UpperCI']   = np.exp(result_df['UpperCI'])
        result_df = result_df[[
            'Variable', 'Coefficient', 'OddsRatio',
            'std_error', 'z_value', 'LowerCI', 'UpperCI', 'p-value',
        ]]
    else:
        # GLM with log-link: add MeanRatio
        if isinstance(fitted_model.family.link, sm.families.links.Log):
            result_df['MeanRatio'] = np.exp(result_df['Coefficient'])
            result_df['LowerCI']   = np.exp(result_df['LowerCI'])
            result_df['UpperCI']   = np.exp(result_df['UpperCI'])
            result_df = result_df[[
                'Variable', 'Coefficient', 'MeanRatio',
                'std_error', 'z_value', 'LowerCI', 'UpperCI', 'p-value',
            ]]
        else:
            result_df = result_df[[
                'Variable', 'Coefficient',
                'std_error', 'z_value', 'LowerCI', 'UpperCI', 'p-value',
            ]]

    # Apply labels
    if labels:
        result_df['Variable'] = result_df['Variable'].apply(
            lambda x: _parse_variable_name(x, labels)
        )

    # Filter intercept/const
    result_df = result_df[
        ~result_df['Variable'].str.lower().isin(
            ['intercept', 'const', 'constant']
        )
    ].reset_index(drop=True)

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
            "calibration_curve": self._calibration_plot,
            "binned_residuals": self._binned_residuals,
        }

    def _binned_residuals(self, backend="plotly"):
        """Generate binned residual plot."""
        from isaric.visualization.lineplots import binned_residuals_plot
        from isaric.modelevaluation.assumptions import test_binned_residuals

        y_prob = self.fitted_model.predict(self.X)
        result = test_binned_residuals(self.y, y_prob)

        return binned_residuals_plot(
            bin_centers=result['bin_centers'],
            mean_residuals=result['mean_residuals'],
            lower_bound=result['lower_bound'],
            upper_bound=result['upper_bound'],
            n_per_bin=result['n_per_bin'],
            title="Binned Residual Plot - Logistic Regression",
            backend=backend
        )

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

    def _forest_plot(self, backend="plotly"):
        """Generate forest plot for Odds Ratios."""
        from isaric.visualization.forestplots import odds_ratio_plot
        return odds_ratio_plot(
            self.result_df,
            effect_col='OddsRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Odds Ratios (Logistic Regression)",
            backend=backend
        )

    def _confusion_matrix(self, backend="plotly"):
        """Generate confusion matrix heatmap."""
        from isaric.visualization.heatmaps import confusion_matrix_heatmap
        y_prob = self.fitted_model.fittedvalues
        y_pred = (y_prob >= 0.5).astype(int)
        return confusion_matrix_heatmap(
            y_true=self.y,
            y_pred=y_pred,
            class_names=['Negative', 'Positive'],
            title="Confusion Matrix - Logistic Regression",
            backend=backend
        )

    def _calibration_plot(self, backend="plotly"):
        """Generate calibration curve plot."""
        from isaric.visualization.lineplots import line_with_ci
        from isaric.modelevaluation.calibration import calibration_curve
        import pandas as pd
        
        y_prob = self.fitted_model.predict(self.X)
        calib = calibration_curve(self.y, y_prob)
        
        data = pd.DataFrame({
            'predicted': calib['mean_predicted'],
            'observed': calib['fraction_positive'],
            'ci_lower': calib['fraction_positive'] * 0.9,
            'ci_upper': calib['fraction_positive'] * 1.1,
        })
        
        return line_with_ci(
            data=data,
            x_col='predicted',
            y_col='observed',
            ci_lower_col='ci_lower',
            ci_upper_col='ci_upper',
            title="Calibration Curve - Logistic Regression",
            xaxis_title="Predicted Probability",
            yaxis_title="Observed Proportion",
            backend=backend
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """Train the logistic regression model."""
        return self._model.fit()

    def _build_result_df(self):
        """Build results DataFrame with Odds Ratios."""
        return _build_result_df(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            labels=self.labels,
            is_logistic=True
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate classification metrics."""
        from isaric.modelevaluation.metrics import compute_classification_metrics

        y_prob = self.fitted_model.predict(self.X)
        y_pred = (y_prob >= 0.5).astype(int)

        result = compute_classification_metrics(
            self.y, y_pred, y_prob=y_prob
        )

        # ============ MODEL INFO ============
        import numpy as np
        info = {
            "model_type": self.model_type,
            "framework": "statsmodels",
            "method": "Binomial GLM",
            "link_function": type(self.fitted_model.family.link).__name__,
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_events": int(self.y.sum()),
            "N_non_events": int((1 - self.y).sum()),
            "N_features": self.X.shape[1],
            "parameters": {
                "link": self.link if hasattr(self, 'link') else "logit",
            },
        }

        if hasattr(self.fitted_model, 'converged'):
            info["converged"] = bool(self.fitted_model.converged)

        if hasattr(self.fitted_model, 'llf'):
            info["log_likelihood"] = float(self.fitted_model.llf)
        if hasattr(self.fitted_model, 'aic'):
            info["aic"] = float(self.fitted_model.aic)
        if hasattr(self.fitted_model, 'bic'):
            info["bic"] = float(self.fitted_model.bic)

        try:
            ll_model = self.fitted_model.llf
            ll_null = self.fitted_model.llnull
            info["mcfadden_r2"] = float(1 - (ll_model / ll_null))
        except Exception:
            pass

        try:
            y_array = np.asarray(self.y).ravel()
            fitted = np.asarray(self.fitted_model.fittedvalues).ravel()
            y_mean = y_array.mean()
            efron = 1 - (
                np.sum((y_array - fitted) ** 2) /
                np.sum((y_array - y_mean) ** 2)
            )
            info["efron_r2"] = float(efron)
        except Exception:
            pass

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Execute k-fold cross-validation."""
        from isaric.modelevaluation.crossvalidation import (
            kfold_cross_validation,
            repeated_kfold_cross_validation
        )
        
        if repetitions > 1:
            return repeated_kfold_cross_validation(
                self._model, self.X, self.y,
                n_splits=k_folds,
                n_repeats=repetitions,
                scoring='roc_auc'
            )
        else:
            return kfold_cross_validation(
                self._model, self.X, self.y,
                n_splits=k_folds,
                scoring='roc_auc'
            )

    def _calibration_curve(self):
        """Generate calibration curve."""
        from isaric.modelevaluation.calibration import (
            compute_brier_score,
            calibration_curve,
            binned_calibration
        )
        
        y_prob = self.fitted_model.predict(self.X)
        
        return {
            'brier_score': compute_brier_score(self.y, y_prob),
            'calibration_curve': calibration_curve(self.y, y_prob),
            'calibration_table': binned_calibration(self.y, y_prob)
        }

    def _check_assumptions(self):
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_vif,
            test_epv,
            test_binned_residuals
        )

        # VIF
        vif_result = test_vif(self.X)
        # EPV
        epv_result = test_epv(self.y, n_predictors=len(self.X.columns))
        # Binned residuals
        y_prob = self.fitted_model.predict(self.X)
        binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'vif': vif_result.to_dict(orient='records'),
            'epv': epv_result,
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        """Split data into train/test sets."""
        from isaric.modelevaluation.traintest import stratified_holdout
        
        data = pd.concat([self.X, self.y], axis=1)
        return stratified_holdout(
            data,
            target_col=self.dependent_var,
            test_size=test_size
        )

    def _validate_external(self, external_data):
        """Validate logistic regression on external independent dataset."""
        from isaric.validation.external import validate_classification
        return validate_classification(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            add_const=True,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for log-OR coefficients.

        Uses bootstrap_coefficients() with the type configured in fit()
        via bootstrap_type= (default bias_corrected for regression).
        Excludes intercept/const from results.

        Returns:
            Dictionary with:
            - 'coefficients': DataFrame with variable, observed (log-OR),
              bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
              bootstrap_p_value, bootstrap_type, n_iterations.
            - 'bootstrap_type': type used.
            - 'n_iterations': number of iterations.
        """
        import statsmodels.api as sm
        from isaric.modelevaluation.bootstrap import bootstrap_coefficients

        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        family = self.fitted_model.family

        def fit_func(X_boot, y_boot):
            try:
                model = sm.GLM(
                    endog=y_boot,
                    exog=X_boot,
                    family=family
                )
                result = model.fit(disp=False)
                return result.params.values
            except Exception:
                return np.full(X_boot.shape[1], np.nan)

        # Exclude intercept/const from variable names
        var_names = [
            v for v in self.X.columns
            if v.lower() not in ('intercept', 'const', 'constant')
        ]
        X_no_const = self.X[[v for v in self.X.columns if v.lower() not in ('intercept', 'const', 'constant')]]

        # Re-run fit_func with intercept (statsmodels needs const in X)
        # but label only the non-intercept variables
        def fit_func_labeled(X_boot, y_boot):
            try:
                model = sm.GLM(
                    endog=y_boot,
                    exog=X_boot,
                    family=family
                )
                result = model.fit(disp=False)
                params = result.params
                coefs = np.array([
                    params[v] for v in self.X.columns
                    if v.lower() not in ('intercept', 'const', 'constant')
                ])
                metrics = {}
                if hasattr(result, 'aic'):
                    metrics['aic'] = float(result.aic)
                if hasattr(result, 'bic'):
                    metrics['bic'] = float(result.bic)
                if hasattr(result, 'llf'):
                    metrics['log_likelihood'] = float(result.llf)
                if hasattr(result, 'deviance'):
                    metrics['deviance'] = float(result.deviance)
                return (coefs, metrics)
            except Exception:
                return np.full(len(var_names), np.nan)

        result = bootstrap_coefficients(
            X=self.X,
            y=self.y,
            fit_func=fit_func_labeled,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            variable_names=var_names,
            random_state=getattr(self, '_random_seed', 42),
            collect_metrics=True,
        )

        if isinstance(result, tuple):
            coef_df, metrics_df = result
        else:
            coef_df, metrics_df = result, None

        return {
            'coefficients':        coef_df,
            'bootstrapped_metrics': metrics_df,
            'bootstrap_type':      bootstrap_type,
            'n_iterations':        n_iterations,
        }

        df = bootstrap_coefficients(
            X=self.X,
            y=self.y,
            fit_func=fit_func_labeled,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            variable_names=var_names,
            random_state=getattr(self, '_random_seed', 42),
        )

        return {
            'coefficients': df,
            'bootstrap_type': bootstrap_type,
            'n_iterations': n_iterations,
        }

    def _validate_sensitivity(self):
        """Sensitivity analysis."""
        from isaric.validation.sensitivity import alternative_missing_handling
        
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(
            data,
            self.dependent_var,
            self.independent_vars
        )

    def _validate_subgroups(self, subgroups):
        """Subgroup analysis."""
        from isaric.validation.subgroup import stratified_metrics
        
        return stratified_metrics(
            self.fitted_model,
            self.X,
            self.y,
            subgroups
        )

    def _validate_net_benefit(self):
        """Decision curve analysis."""
        from isaric.validation.netprofit import decision_curve_analysis
        
        y_prob = self.fitted_model.predict(self.X)  # ← usar predict, não predict_proba
        return decision_curve_analysis(self.y, y_prob)


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
            "residuals_plot": self._residuals_plot,
            "qq_plot": self._qq_plot,
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

    def _forest_plot(self, backend="plotly"):
        """Generate forest plot for coefficients."""
        from isaric.visualization.forestplots import coefficient_plot
        return coefficient_plot(
            self.result_df,
            effect_col='Coefficient',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Coefficients (GLM)",
            backend=backend
        )

    def _residuals_plot(self, backend="plotly"):
        """Generate residuals vs fitted plot."""
        from isaric.visualization.lineplots import multi_line_plot
        import pandas as pd
        
        y_pred = self.fitted_model.predict(self.X)
        residuals = self.y - y_pred
        
        data = pd.DataFrame({
            'fitted': y_pred,
            'residuals': residuals
        })
        
        return multi_line_plot(
            data=data,
            x_col='fitted',
            y_cols=['residuals'],
            title="Residuals vs Fitted - GLM",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            backend=backend
        )

    def _qq_plot(self, backend="plotly"):
        """Generate Q-Q plot for normality."""
        from isaric.visualization.lineplots import multi_line_plot
        from isaric.modelevaluation.calibration import qq_plot as calc_qq
        import pandas as pd
        
        residuals = self.y - self.fitted_model.predict(self.X)
        qq_data = calc_qq(residuals)
        
        data = pd.DataFrame({
            'theoretical': qq_data['theoretical_quantiles'],
            'sample': qq_data['sample_quantiles']
        })
        
        return multi_line_plot(
            data=data,
            x_col='theoretical',
            y_cols=['sample'],
            title="Q-Q Plot - GLM",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            backend=backend
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """Train the GLM model."""
        return self._model.fit()

    def _build_result_df(self):
        """Build results DataFrame with coefficients."""
        return _build_result_df(
            fitted_model=self.fitted_model,
            X=self.X,
            y=self.y,
            labels=self.labels,
            is_logistic=False
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate regression metrics."""
        from isaric.modelevaluation.metrics import (
            compute_regression_metrics,
            compute_information_criteria,
            compute_pseudo_r2
        )

        y_pred = self.fitted_model.predict(self.X)
        result = compute_regression_metrics(self.y, y_pred)

        if hasattr(self.fitted_model, 'llf'):
            info_criteria = compute_information_criteria(
                self.fitted_model,
                n_samples=len(self.y),
                n_params=len(self.X.columns)
            )
            result.update(info_criteria)

            pseudo_r2 = compute_pseudo_r2(
                self.fitted_model,
                self.y,
                y_pred
            )
            result.update(pseudo_r2)

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "statsmodels",
            "method": "Generalized Linear Model",
            "family": type(self.fitted_model.family).__name__,
            "link_function": type(self.fitted_model.family.link).__name__,
            "random_seed": self._random_seed,
            "N_observations": len(self.y),
            "N_features": self.X.shape[1],
            "parameters": {
                "family": getattr(self, 'family', 'gaussian'),
                "link": getattr(self, 'link', 'identity'),
            },
        }

        if hasattr(self.fitted_model, 'converged'):
            info["converged"] = bool(self.fitted_model.converged)
        if hasattr(self.fitted_model, 'llf'):
            info["log_likelihood"] = float(self.fitted_model.llf)
        if hasattr(self.fitted_model, 'aic'):
            info["aic"] = float(self.fitted_model.aic)
        if hasattr(self.fitted_model, 'bic'):
            info["bic"] = float(self.fitted_model.bic)
        if hasattr(self.fitted_model, 'deviance'):
            info["deviance"] = float(self.fitted_model.deviance)

        self.model_info = info

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Execute k-fold cross-validation."""
        from isaric.modelevaluation.crossvalidation import (
            kfold_cross_validation,
            repeated_kfold_cross_validation
        )
        
        if repetitions > 1:
            return repeated_kfold_cross_validation(
                self._model, self.X, self.y,
                n_splits=k_folds,
                n_repeats=repetitions,
                scoring='neg_mean_squared_error'
            )
        else:
            return kfold_cross_validation(
                self._model, self.X, self.y,
                n_splits=k_folds,
                scoring='neg_mean_squared_error'
            )

    def _calibration_curve(self):
        """Generate calibration diagnostics."""
        from isaric.modelevaluation.calibration import (
            residuals_vs_fitted,
            qq_plot
        )
        
        y_pred = self.fitted_model.predict(self.X)
        residuals = self.y - y_pred
        
        return {
            'residuals_fitted': residuals_vs_fitted(residuals, y_pred),
            'qq_plot': qq_plot(residuals)
        }

    def _check_assumptions(self):
        """Check model assumptions (VIF + binned residuals)."""
        from isaric.modelevaluation.assumptions import (
            test_durbin_watson,
            test_shapiro_wilk,
            test_vif,
            test_cooks_distance,
            test_binned_residuals
        )

        residuals = self.y - self.fitted_model.predict(self.X)

        # Binned residuals (para modelos com família binomial)
        binned_result = None
        if isinstance(self.fitted_model.family, sm.families.Binomial):
            y_prob = self.fitted_model.predict(self.X)
            binned_result = test_binned_residuals(self.y, y_prob)

        return {
            'durbin_watson': test_durbin_watson(residuals),
            'shapiro_wilk': test_shapiro_wilk(residuals),
            'vif': test_vif(self.X).to_dict(orient='records'),
            'cooks_distance': test_cooks_distance(
                self.fitted_model, self.X, self.y
            ),
            'binned_residuals': binned_result,
        }

    def _train_test_split(self, test_size=0.2):
        """Split data into train/test sets."""
        from isaric.modelevaluation.traintest import holdout_validation
        
        data = pd.concat([self.X, self.y], axis=1)
        return holdout_validation(data, test_size=test_size)

    def _validate_external(self, external_data):
        """Validate GLM on external independent dataset."""
        from isaric.validation.external import validate_classification
        is_binomial = isinstance(
            self.fitted_model.family, __import__('statsmodels').api.families.Binomial
        )
        return validate_classification(
            fitted_model=self.fitted_model,
            external_data=external_data,
            dependent_var=self.dependent_var,
            independent_vars=self.independent_vars,
            add_const=True,
        ) if is_binomial else {
            'note': 'External validation for non-binomial GLM not yet implemented.'
        }

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap confidence intervals and p-values for GLM coefficients.

        Uses bootstrap_coefficients() with the type configured in fit()
        via bootstrap_type= (default bias_corrected for regression).
        Excludes intercept/const from results.

        Returns:
            Dictionary with:
            - 'coefficients': DataFrame with variable, observed,
              bootstrap_mean, bootstrap_se, ci_lower, ci_upper,
              bootstrap_p_value, bootstrap_type, n_iterations.
            - 'bootstrap_type': type used.
            - 'n_iterations': number of iterations.
        """
        import statsmodels.api as sm
        from isaric.modelevaluation.bootstrap import bootstrap_coefficients

        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        family = self.fitted_model.family

        var_names = [
            v for v in self.X.columns
            if v.lower() not in ('intercept', 'const', 'constant')
        ]

        def fit_func_labeled(X_boot, y_boot):
            try:
                model = sm.GLM(
                    endog=y_boot,
                    exog=X_boot,
                    family=family
                )
                result = model.fit(disp=False)
                params = result.params
                coefs = np.array([
                    params[v] for v in self.X.columns
                    if v.lower() not in ('intercept', 'const', 'constant')
                ])
                metrics = {}
                if hasattr(result, 'aic'):
                    metrics['aic'] = float(result.aic)
                if hasattr(result, 'bic'):
                    metrics['bic'] = float(result.bic)
                if hasattr(result, 'llf'):
                    metrics['log_likelihood'] = float(result.llf)
                if hasattr(result, 'deviance'):
                    metrics['deviance'] = float(result.deviance)
                return (coefs, metrics)
            except Exception:
                return np.full(len(var_names), np.nan)

        result = bootstrap_coefficients(
            X=self.X,
            y=self.y,
            fit_func=fit_func_labeled,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            variable_names=var_names,
            random_state=getattr(self, '_random_seed', 42),
            collect_metrics=True,
        )

        if isinstance(result, tuple):
            coef_df, metrics_df = result
        else:
            coef_df, metrics_df = result, None

        return {
            'coefficients':        coef_df,
            'bootstrapped_metrics': metrics_df,
            'bootstrap_type':      bootstrap_type,
            'n_iterations':        n_iterations,
        }

        df = bootstrap_coefficients(
            X=self.X,
            y=self.y,
            fit_func=fit_func_labeled,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            variable_names=var_names,
            random_state=getattr(self, '_random_seed', 42),
        )

        return {
            'coefficients': df,
            'bootstrap_type': bootstrap_type,
            'n_iterations': n_iterations,
        }

    def _validate_sensitivity(self):
        """Sensitivity analysis."""
        from isaric.validation.sensitivity import alternative_missing_handling
        
        data = pd.concat([self.X, self.y], axis=1)
        return alternative_missing_handling(
            data,
            self.dependent_var,
            self.independent_vars
        )

    def _validate_subgroups(self, subgroups):
        """Subgroup analysis."""
        from isaric.validation.subgroup import stratified_metrics
        
        return stratified_metrics(
            self.fitted_model,
            self.X,
            self.y,
            subgroups
        )

    def _validate_net_benefit(self):
        """Not applicable for GLM regression."""
        return None
    