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

    Args:
        data: Input DataFrame.
        duration_var: Time-to-event column.
        event_var: Event indicator column.
        independent_vars: Predictor variable names.
        formula: Patsy-style formula (optional).

    Returns:
        DataFrame with duration, event, and predictor columns.
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

    Args:
        fitted_model: Fitted CoxPHFitter.
        labels: Dictionary for variable display labels.

    Returns:
        DataFrame with HazardRatio, LowerCI, UpperCI, p-value.
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

        Args:
            model: Configured CoxPHFitter (from create_survival_model).
            model_data: DataFrame for training.
            duration_var: Time-to-event column.
            event_var: Event indicator column.
            independent_vars: Predictor variable names.
            labels: Dictionary for variable display labels.
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

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "survival_cox").
            duration_var: Time-to-event column.
            event_var: Event indicator column.
            independent_vars: Predictor variable names.
            formula: Patsy-style formula (optional).
            penalizer: L2 regularization strength.
            labels: Dictionary for variable display labels.

        Returns:
            SurvivalCox instance ready for training.
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

    def _forest_plot(self):
        """Generate forest plot for Hazard Ratios."""
        from isaric.visualization.forestplots import hazard_ratio_plot

        fig = hazard_ratio_plot(
            self.result_df,
            effect_col='HazardRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot - Hazard Ratios (Cox PH)"
        )
        return fig

    def _survival_curve(self):
        """Generate baseline survival curve."""
        from isaric.visualization.survivalcurves import baseline_survival_curve

        fig = baseline_survival_curve(
            self.fitted_model,
            title="Baseline Survival Curve (Cox Model)"
        )
        return fig


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

        Args:
            model: Configured KaplanMeierFitter.
            model_data: DataFrame with duration and event columns.
            duration_var: Time-to-event column.
            event_var: Event indicator column.
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

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "survival_km").
            duration_var: Time-to-event column.
            event_var: Event indicator column.

        Returns:
            KaplanMeier instance ready for training.
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

    def _survival_curve(self):
        """Generate Kaplan-Meier survival curve."""
        from isaric.visualization.survivalcurves import kaplan_meier_curve

        fig = kaplan_meier_curve(
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            title="Kaplan-Meier Survival Curve"
        )
        return fig