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
    duration_var: str = "AVAL",
    event_var: Optional[str] = None,
    censoring_var: Optional[str] = None,
    independent_vars: Optional[List[str]] = None,
    formula: Optional[str] = None,
    avalu_var: str = "AVALU",
    penalizer: float = 0.1
) -> Tuple[CoxPHFitter, pd.DataFrame, str, Optional[str]]:
    """
    Configure a Cox Proportional Hazards model (not fitted).

    Supports:
    - Matrix-based approach (independent_vars)
    - Formula-based approach (formula)
    - CDISC ADTTE conventions (AVAL, CNSR, TRTA, AVALU)

    If formula is not provided, it's built from independent_vars:
        formula = f"{duration_var} ~ {' + '.join(independent_vars)}"

    Args:
        data: Input DataFrame in ARC format.
        duration_var: Time-to-event column (default: "AVAL").
        event_var: Event indicator column. Computed from censoring_var if None.
        censoring_var: Censoring indicator column (1=censored).
        independent_vars: Predictor variable names (used to build formula).
        formula: Patsy-style formula (overrides independent_vars).
        avalu_var: Time unit column (default: "AVALU").
        penalizer: L2 regularization strength (default 0.1).

    Returns:
        Tuple of (model, model_data, time_unit, formula).
    """
    if penalizer < 0:
        raise ValueError(f"penalizer must be non-negative. Received: {penalizer}")

    # Computa event_var a partir de censoring_var
    if censoring_var is not None and event_var is None:
        if censoring_var not in data.columns:
            raise ValueError(f"Column '{censoring_var}' not found in DataFrame.")
        event_var = "_event_computed"
        data = data.copy()
        data[event_var] = 1 - data[censoring_var]

    if duration_var is None:
        duration_var = "AVAL"
    if event_var is None:
        event_var = "event"

    # Constrói formula se não fornecida
    if formula is None and independent_vars:
        formula = f"{duration_var} ~ {' + '.join(independent_vars)}"

    # Extrai unidade de tempo
    time_unit = "Time"
    if avalu_var and avalu_var in data.columns:
        units = data[avalu_var].dropna().unique()
        if len(units) > 0:
            time_unit = str(units[0])

    # Prepara model_data
    if formula is not None:
        # Formula-based: mantém todas as colunas necessárias
        required_cols = [duration_var, event_var]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        model_data = data.dropna(subset=required_cols).copy()
    else:
        model_data = _prepare_model_data(
            data, duration_var, event_var, independent_vars or []
        )

    model = CoxPHFitter(penalizer=penalizer)

    return model, model_data, time_unit, formula


def create_kaplan_meier_model(
    data: pd.DataFrame,
    duration_var: str = "AVAL",
    event_var: Optional[str] = None,
    censoring_var: Optional[str] = None,
    avalu_var: str = "AVALU"
) -> Tuple[KaplanMeierFitter, pd.DataFrame, str]:
    """
    Configure a Kaplan-Meier model (not fitted).

    Supports CDISC ADTTE conventions.

    Returns:
        Tuple of (model, model_data, time_unit).
    """
    # Se censoring_var fornecido e event_var não, computa event_var
    if censoring_var is not None and event_var is None:
        if censoring_var not in data.columns:
            raise ValueError(f"Column '{censoring_var}' not found in DataFrame.")
        event_var = "_event_computed"
        data = data.copy()
        data[event_var] = 1 - data[censoring_var]

    for col in [duration_var, event_var]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Extrai unidade de tempo
    time_unit = "Time"
    if avalu_var and avalu_var in data.columns:
        units = data[avalu_var].dropna().unique()
        if len(units) > 0:
            time_unit = str(units[0])

    model_data = data[[duration_var, event_var]].dropna().copy()
    model = KaplanMeierFitter()

    return model, model_data, time_unit


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
    time_unit: str = "Time",
    formula: Optional[str] = None,
    target_time: float = 365,
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
        self.time_unit = time_unit
        self.formula = formula
        self.target_time = target_time
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
            "schoenfeld_residuals": self._schoenfeld_residuals,
            "martingale_residuals": self._martingale_residuals,
            "deviance_residuals": self._deviance_residuals,
            "brier_score": self._brier_score,
            "roc_time_dependent": self._roc_time_dependent,
            "calibration_plot": self._calibration_plot,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "survival_cox",
        duration_var: str = "AVAL",
        event_var: Optional[str] = None,
        target_time: float = 365,
        censoring_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        avalu_var: str = "AVALU",
        penalizer: float = 0.1,
        labels: Optional[Dict[str, str]] = None,
        **params
    ) -> "SurvivalCox":
        model_config, model_data, time_unit, formula = create_survival_model(
            data=data,
            duration_var=duration_var,
            event_var=event_var,
            censoring_var=censoring_var,
            independent_vars=independent_vars,
            formula=formula,
            avalu_var=avalu_var,
            penalizer=penalizer
        )

        return cls(
            model=model_config,
            model_data=model_data,
            duration_var=duration_var,
            event_var=event_var,
            independent_vars=independent_vars or [],
            labels=labels,
            time_unit=time_unit,
            formula=formula,
            target_time=target_time, 
            **params
        )
    
    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """Train the Cox PH model."""
        if self.formula is not None:
            # Formula-based fit (lifelines lida com categóricos)
            return self._model.fit(
                self.model_data,
                formula=self.formula,
                event_col=self.event_var
            )
        else:
            # Matrix-based fit
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
        """
        Cross-validation for Cox model using Concordance Index.

        Performs k-fold cross-validation manually because lifelines
        CoxPHFitter does not follow sklearn's estimator API.
        """
        from sklearn.model_selection import KFold
        from lifelines import CoxPHFitter
        import numpy as np

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_c_indices = []

        for train_idx, test_idx in kf.split(self.model_data):
            train_df = self.model_data.iloc[train_idx]
            test_df = self.model_data.iloc[test_idx]

            model_fold = CoxPHFitter(penalizer=self._model.penalizer)
            model_fold.fit(
                train_df,
                duration_col=self.duration_var,
                event_col=self.event_var
            )

            # C-index no fold de teste
            c_index = model_fold.score(
                test_df,
                scoring_method="concordance_index"
            )
            cv_c_indices.append(c_index)

        return {
            'scores': np.array(cv_c_indices),
            'mean_score': float(np.mean(cv_c_indices)),
            'std_score': float(np.std(cv_c_indices))
        }

    def _calibration_curve(self):
        """Survival calibration, Brier Score, and ROC."""
        from isaric.modelevaluation.calibration import (
            survival_calibration,
            survival_brier_score,
            survival_roc
        )

        result = {}

        try:
            result['calibration'] = survival_calibration(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time
            )
        except Exception:
            pass

        try:
            result['brier_score'] = survival_brier_score(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time
            )
        except Exception:
            pass

        try:
            result['roc'] = survival_roc(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time
            )
        except Exception:
            pass

        return result

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
        """Bootstrap validation for Cox model using C-index."""
        from isaric.validation.bootstrap import bootstrap_metrics
        from lifelines.utils import concordance_index

        def c_index_metric(y_true, y_pred):
            # y_true = event indicator (0/1), y_pred = partial hazard
            # Para C-index: usa duration_var como tempo
            return concordance_index(
                self.model_data[self.duration_var],
                -y_pred,  # Negativo: risco alto = sobrevida curta
                y_true
            )

        return bootstrap_metrics(
            self.fitted_model,
            self.X,
            self.y,
            n_iterations=n_iterations,
            metric_func=c_index_metric,
            prediction_method='predict_partial_hazard'
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
        """Generate baseline survival curve with time unit."""
        from isaric.visualization.survivalcurves import baseline_survival_curve

        return baseline_survival_curve(
            self.fitted_model,
            title="Baseline Survival Curve (Cox Model)",
            xaxis_title=f"Time ({self.time_unit})",
            backend=backend
        )

    def _schoenfeld_residuals(self, backend="plotly"):
        """Generate Schoenfeld residuals plots for all covariates."""
        from isaric.visualization.survivalcurves import schoenfeld_residuals_plot

        residuals = self.fitted_model.compute_residuals(
            self.model_data, 'schoenfeld'
        )

        figs = []
        for col in residuals.columns:
            fig = schoenfeld_residuals_plot(
                residuals=residuals[col].values,
                times=residuals.index.values,
                covariate_name=col,
                backend=backend
            )
            figs.append(fig)

        return figs  # Retorna lista de figuras (uma por covariável)


    def _martingale_residuals(self, backend="plotly"):
        """Generate Martingale residuals plots for all covariates."""
        from isaric.visualization.survivalcurves import martingale_residuals_plot

        residuals = self.fitted_model.compute_residuals(
            self.model_data, 'martingale'
        )

        # Covariáveis (excluindo duration e event)
        cols_to_plot = [
            c for c in self.model_data.columns
            if c not in [self.duration_var, self.event_var]
        ]

        figs = []
        for col in cols_to_plot:
            fig = martingale_residuals_plot(
                residuals=residuals['martingale'].values,
                covariate=self.model_data[col].values,
                covariate_name=col,
                add_smoother=True,
                backend=backend
            )
            figs.append(fig)

        return figs


    def _deviance_residuals(self, backend="plotly"):
        """Generate Deviance residuals plots for all covariates."""
        from isaric.visualization.survivalcurves import deviance_residuals_plot

        residuals = self.fitted_model.compute_residuals(
            self.model_data, 'deviance'
        )

        cols_to_plot = [
            c for c in self.model_data.columns
            if c not in [self.duration_var, self.event_var]
        ]

        figs = []
        for col in cols_to_plot:
            fig = deviance_residuals_plot(
                residuals=residuals['deviance'].values,
                covariate=self.model_data[col].values,
                covariate_name=col,
                backend=backend
            )
            figs.append(fig)

        return figs

    def _brier_score(self, backend="plotly"):
        """Generate Brier Score plot."""
        from isaric.visualization.survivalcurves import brier_score_plot
        from isaric.modelevaluation.calibration import survival_brier_score

        data = survival_brier_score(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time
        )
        return brier_score_plot(
            time_points=data['time_points'],
            brier_scores=data['brier_scores'],
            target_time=self.target_time,
            backend=backend
        )


    def _roc_time_dependent(self, backend="plotly"):
        """Generate time-dependent ROC plot."""
        from isaric.visualization.survivalcurves import roc_time_dependent_plot
        from isaric.modelevaluation.calibration import survival_roc

        data = survival_roc(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time
        )
        return roc_time_dependent_plot(
            fpr=data['fpr'],
            tpr=data['tpr'],
            auc=data['auc'],
            target_time=self.target_time,
            backend=backend
        )


    def _calibration_plot(self, backend="plotly"):
        """Generate calibration plot for survival."""
        from isaric.visualization.survivalcurves import calibration_plot
        from isaric.modelevaluation.calibration import survival_calibration

        data = survival_calibration(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time
        )
        return calibration_plot(
            predicted_survival=data['predicted_survival'],
            observed_survival=data['observed_survival'],
            target_time=self.target_time,
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
        time_unit: str = "Time",
        **kwargs
    ):
        """
        Initialize KaplanMeier with configured model and data.
        """
        self._model = model
        self.model_data = model_data
        self.duration_var = duration_var
        self.event_var = event_var
        self.time_unit = time_unit
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
            "schoenfeld_residuals": self._schoenfeld_residuals,
            "martingale_residuals": self._martingale_residuals,
            "deviance_residuals": self._deviance_residuals,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "survival_km",
        duration_var: str = "AVAL",
        event_var: Optional[str] = None,
        censoring_var: Optional[str] = None,
        avalu_var: str = "AVALU",
        **params
    ) -> "KaplanMeier":
        model_config, model_data, time_unit = create_kaplan_meier_model(
            data=data,
            duration_var=duration_var,
            event_var=event_var,
            censoring_var=censoring_var,
            avalu_var=avalu_var
        )
        return cls(
            model=model_config,
            model_data=model_data,
            duration_var=duration_var,
            event_var=event_var,
            time_unit=time_unit,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
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
        """Bootstrap validation for Kaplan-Meier median survival."""
        from isaric.validation.bootstrap import bootstrap_metrics
        from sklearn.metrics import mean_squared_error

        # KaplanMeier não tem predict - usamos o median_survival_time_
        # como métrica única
        import numpy as np
        from lifelines import KaplanMeierFitter

        rng = np.random.default_rng(42)
        n_samples = len(self.model_data)
        bootstrap_values = []

        for _ in range(n_iterations):
            indices = rng.integers(0, n_samples, n_samples)
            data_boot = self.model_data.iloc[indices]

            kmf = KaplanMeierFitter()
            kmf.fit(
                data_boot[self.duration_var],
                event_observed=data_boot[self.event_var]
            )
            bootstrap_values.append(kmf.median_survival_time_)

        values = np.array(bootstrap_values)

        from isaric.validation.bootstrap import confidence_interval
        ci_lower, ci_upper = confidence_interval(values)

        return {
            'values': values,
            'mean': float(np.mean(values)),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

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
        """Generate Kaplan-Meier survival curve with time unit."""
        from isaric.visualization.survivalcurves import kaplan_meier_curve
        return kaplan_meier_curve(
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            title="Kaplan-Meier Survival Curve",
            xaxis_title=f"Time ({self.time_unit})",
            backend=backend
        )