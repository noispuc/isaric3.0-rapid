"""
Survival analysis for the RAPID methodology.

This module provides functions to configure survival models and
concrete pipeline classes for Cox Proportional Hazards and
Kaplan-Meier analysis.

Functions (Configuration):
- create_survival_model: Configure a Cox PH model (not fitted).
- create_kaplan_meier_model: Configure a Kaplan-Meier model (not fitted).

Subclasses (Pipelines):
- SurvivalCox: Concrete pipeline for Cox Proportional Hazards.
- KaplanMeier: Concrete pipeline for Kaplan-Meier survival analysis.

Helper Functions:
- _prepare_model_data: Prepare data for CoxPHFitter.
- _build_result_df: Build Hazard Ratios DataFrame.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from lifelines import CoxPHFitter, KaplanMeierFitter
from isaric.rapid import RAPID


# ============================================================================
# PUBLIC FUNCTIONS (CONFIGURATION)
# ============================================================================

def create_survival_model(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    independent_vars: List[str],
    formula: Optional[str] = None,
    penalizer: float = 0.1
) -> Tuple[CoxPHFitter, pd.DataFrame]:
    """
    Configure a Cox Proportional Hazards model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).
        independent_vars: Predictor variable names.
        formula: Patsy-style formula (optional).
        penalizer: L2 regularization strength (default 0.1).

    Returns:
        Tuple of (model, model_data).

    Raises:
        ValueError: If columns are not found or penalizer is invalid.
    """
    if penalizer < 0:
        raise ValueError(f"penalizer must be non-negative. Received: {penalizer}")

    model_data = _prepare_model_data(
        data, duration_var, event_var, independent_vars, formula
    )

    model = CoxPHFitter(penalizer=penalizer)

    return model, model_data


def create_kaplan_meier_model(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str
) -> Tuple[KaplanMeierFitter, pd.DataFrame]:
    """
    Configure a Kaplan-Meier model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).

    Returns:
        Tuple of (model, model_data).

    Raises:
        ValueError: If columns are not found.
    """
    for col in [duration_var, event_var]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    model_data = data[[duration_var, event_var]].dropna().copy()
    model = KaplanMeierFitter()

    return model, model_data


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _prepare_model_data(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    independent_vars: List[str],
    formula: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare model data for CoxPHFitter.
    """
    if formula:
        required_cols = [duration_var, event_var]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        return data.dropna(subset=required_cols).copy()

    required_cols = [duration_var, event_var] + independent_vars
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    model_data = data[required_cols].dropna().copy()
    return model_data


def _build_result_df(
    fitted_model: CoxPHFitter,
    labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Build Hazard Ratios DataFrame from fitted Cox model.
    """
    summary = fitted_model.summary.copy()

    summary['HazardRatio'] = np.exp(summary['coef'])
    summary['LowerCI'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['UpperCI'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    summary['p-value'] = summary['p'].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    result_df = summary[['HazardRatio', 'LowerCI', 'UpperCI', 'p-value']].reset_index()
    result_df = result_df.rename(columns={result_df.columns[0]: 'Variable'})

    if labels:
        result_df['Variable'] = result_df['Variable'].map(labels).fillna(
            result_df['Variable']
        )

    return result_df


# ============================================================================
# SUBCLASSES (INHERIT FROM RAPID)
# ============================================================================

class SurvivalCox(RAPID):
    """
    Concrete pipeline for Cox Proportional Hazards.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: CoxPHFitter,
        model_data: pd.DataFrame,
        duration_var: str,
        event_var: str,
        independent_vars: List[str],
        labels: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize SurvivalCox with configured model and data.
        """
        self._model = model
        self.model_data = model_data
        self.duration_var = duration_var
        self.event_var = event_var
        self.independent_vars = independent_vars
        self.labels = labels
        self.model_type = "survival_cox"
        self.X = model_data[independent_vars]
        self.y = model_data[event_var]
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for SurvivalCox."""
        self.plots_map = {
            "forest_plot": self._forest_plot,
            "survival_curve": self._survival_curve,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "survival_cox",
        duration_var: Optional[str] = None,
        event_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        penalizer: float = 0.1,
        labels: Optional[Dict[str, str]] = None,
        **params
    ) -> "SurvivalCox":
        """
        Configure and instantiate the SurvivalCox pipeline.
        """
        model_config, model_data = create_survival_model(
            data=data,
            duration_var=duration_var,
            event_var=event_var,
            independent_vars=independent_vars,
            formula=formula,
            penalizer=penalizer
        )

        return cls(
            model=model_config,
            model_data=model_data,
            duration_var=duration_var,
            event_var=event_var,
            independent_vars=independent_vars,
            labels=labels,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self):
        """Train the Cox PH model."""
        return self._model.fit(
            self.model_data,
            duration_col=self.duration_var,
            event_col=self.event_var
        )

    def _build_result_df(self):
        """Build Hazard Ratios DataFrame."""
        return _build_result_df(
            fitted_model=self.fitted_model,
            labels=self.labels
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate survival metrics."""
        from isaric.modelevaluation.metrics import compute_survival_metrics
        return compute_survival_metrics(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var
        )

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Cross-validation for Cox model."""
        from isaric.modelevaluation.crossvalidation import kfold_cross_validation
        return kfold_cross_validation(
            self._model, self.X, self.y,
            n_splits=k_folds,
            scoring='roc_auc'
        )

    def _calibration_curve(self):
        """Survival calibration."""
        from isaric.modelevaluation.calibration import survival_calibration
        return survival_calibration(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=30
        )

    def _check_assumptions(self):
        """Check Proportional Hazards assumption."""
        from isaric.modelevaluation.assumptions import test_proportional_hazards
        return {
            'proportional_hazards': test_proportional_hazards(
                self.fitted_model,
                self.model_data
            )
        }

    def _train_test_split(self, test_size=0.2):
        """Split data chronologically."""
        from isaric.modelevaluation.traintest import temporal_holdout
        return temporal_holdout(
            self.model_data,
            date_col=self.duration_var,
            test_size=test_size
        )

    def _validate_external(self, external_data):
        """Validate on external dataset."""
        from isaric.validation.external import temporal_validation
        return temporal_validation(
            self.fitted_model,
            external_data,
            dependent_var=self.event_var,
            independent_vars=self.independent_vars
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """Bootstrap validation."""
        from isaric.validation.bootstrap import bootstrap_metrics
        from sklearn.metrics import roc_auc_score
        return bootstrap_metrics(
            self.fitted_model,
            self.X,
            self.y,
            n_iterations=n_iterations,
            metric_func=roc_auc_score
        )

    def _validate_sensitivity(self):
        """Sensitivity analysis."""
        from isaric.validation.sensitivity import alternative_missing_handling
        return alternative_missing_handling(
            self.model_data,
            self.event_var,
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
        """Not applicable for survival."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _forest_plot(self, backend="plotly"):
        """Generate forest plot for Hazard Ratios."""
        from isaric.visualization.forestplots import hazard_ratio_plot

        return hazard_ratio_plot(
            self.result_df,
            effect_col='HazardRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Hazard Ratios (Cox PH)",
            backend=backend
        )

    def _survival_curve(self, backend="plotly"):
        """Generate baseline survival curve."""
        from isaric.visualization.survivalcurves import baseline_survival_curve

        return baseline_survival_curve(
            self.fitted_model,
            title="Baseline Survival Curve (Cox Model)",
            backend=backend
        )


class KaplanMeier(RAPID):
    """
    Concrete pipeline for Kaplan-Meier survival analysis.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: KaplanMeierFitter,
        model_data: pd.DataFrame,
        duration_var: str,
        event_var: str,
        **kwargs
    ):
        """
        Initialize KaplanMeier with configured model and data.
        """
        self._model = model
        self.model_data = model_data
        self.duration_var = duration_var
        self.event_var = event_var
        self.model_type = "survival_km"
        self.X = model_data[[duration_var]]
        self.y = model_data[event_var]
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for KaplanMeier."""
        self.plots_map = {
            "survival_curve": self._survival_curve,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "survival_km",
        duration_var: Optional[str] = None,
        event_var: Optional[str] = None,
        **params
    ) -> "KaplanMeier":
        """
        Configure and instantiate the KaplanMeier pipeline.
        """
        model_config, model_data = create_kaplan_meier_model(
            data=data,
            duration_var=duration_var,
            event_var=event_var
        )

        return cls(
            model=model_config,
            model_data=model_data,
            duration_var=duration_var,
            event_var=event_var,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self):
        """Train the Kaplan-Meier model."""
        return self._model.fit(
            self.model_data[self.duration_var],
            event_observed=self.model_data[self.event_var]
        )

    def _build_result_df(self):
        """Build results DataFrame."""
        return pd.DataFrame({
            'Variable': ['Median_Survival', 'N_Events', 'N_Censored'],
            'Value': [
                self.fitted_model.median_survival_time_,
                self.fitted_model.event_table['observed'].sum(),
                self.fitted_model.event_table['censored'].sum()
            ]
        })

    def _calculate_metrics(self, metrics=None):
        """Calculate survival metrics."""
        return {
            'median_survival': self.fitted_model.median_survival_time_,
            'n_events': int(self.fitted_model.event_table['observed'].sum()),
            'n_censored': int(self.fitted_model.event_table['censored'].sum()),
        }

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Not applicable for Kaplan-Meier."""
        return None

    def _calibration_curve(self):
        """Not applicable for Kaplan-Meier."""
        return None

    def _check_assumptions(self):
        """Not applicable for Kaplan-Meier."""
        return None

    def _train_test_split(self, test_size=0.2):
        """Not applicable for Kaplan-Meier."""
        return None

    def _validate_external(self, external_data):
        """Not applicable for Kaplan-Meier."""
        return None

    def _validate_bootstrap(self, n_iterations=1000):
        """Bootstrap validation for median survival."""
        from isaric.validation.bootstrap import bootstrap_metrics
        from sklearn.metrics import mean_squared_error
        return bootstrap_metrics(
            self.fitted_model,
            self.X,
            self.y,
            n_iterations=n_iterations,
            metric_func=mean_squared_error
        )

    def _validate_sensitivity(self):
        """Not applicable for Kaplan-Meier."""
        return None

    def _validate_subgroups(self, subgroups):
        """Not applicable for Kaplan-Meier."""
        return None

    def _validate_net_benefit(self):
        """Not applicable for Kaplan-Meier."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _survival_curve(self, backend="plotly"):
        """Generate Kaplan-Meier survival curve."""
        from isaric.visualization.survivalcurves import kaplan_meier_curve

        return kaplan_meier_curve(
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            title="Kaplan-Meier Survival Curve",
            backend=backend
        )