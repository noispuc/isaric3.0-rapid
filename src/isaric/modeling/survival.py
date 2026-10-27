"""
Survival analysis for the RAPID methodology.

This module provides the unified Survival pipeline (Cox PH + Kaplan-Meier)
and supporting helper functions.

Functions (Configuration):
- create_survival_model:    Configure a Cox PH model (not fitted).
- create_kaplan_meier_model: Configure a Kaplan-Meier model (not fitted).

Subclasses (Pipeline):
- Survival: Unified pipeline — Cox PH + Kaplan-Meier + optional Log-rank.

Helper Functions:
- _prepare_model_data: Prepare data for CoxPHFitter.
- _build_result_df:    Build Hazard Ratios DataFrame from fitted Cox model.
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
    penalizer: float = 0.1,
    target_time: Optional[float] = None
) -> Tuple[CoxPHFitter, pd.DataFrame, str, Optional[str], str, float]:
    """
    Configure a Cox Proportional Hazards model (not fitted).

    Supports CDISC ADTTE conventions. Handles censoring_var → event_var
    conversion automatically.

    Args:
        data: Input DataFrame.
        duration_var: Time-to-event column (default "AVAL").
        event_var: Event indicator (1=event). Computed from censoring_var
            if not provided.
        censoring_var: Censoring indicator (1=censored, CDISC CNSR).
        independent_vars: Covariate columns for Cox.
        formula: Patsy formula for Cox. If None, built from independent_vars.
        avalu_var: Time unit column (default "AVALU").
        penalizer: Ridge penalizer for CoxPHFitter (default 0.1).
        target_time: Time point for time-dependent metrics. If None,
            uses median of duration_var.

    Returns:
        Tuple of (model, model_data, time_unit, formula, event_var,
        target_time).
    """
    if penalizer < 0:
        raise ValueError(
            f"penalizer must be non-negative. Received: {penalizer}"
        )

    # Compute event_var from censoring_var if needed
    if censoring_var is not None and event_var is None:
        if censoring_var not in data.columns:
            raise ValueError(
                f"Column '{censoring_var}' not found in DataFrame."
            )
        event_var = "_event_computed"
        data = data.copy()
        data[event_var] = 1 - data[censoring_var]

    if duration_var is None:
        duration_var = "AVAL"
    if event_var is None:
        event_var = "event"

    # Build formula if not provided
    if formula is None and independent_vars:
        formula = ' + '.join(independent_vars)

    # Extract time unit
    time_unit = "Time"
    if avalu_var and avalu_var in data.columns:
        units = data[avalu_var].dropna().unique()
        if len(units) > 0:
            time_unit = str(units[0])

    # Prepare model_data
    if formula is not None:
        required_cols = [duration_var, event_var]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        model_data = data.dropna(subset=required_cols).copy()
    else:
        model_data = _prepare_model_data(
            data, duration_var, event_var, independent_vars or []
        )

    # Derive target_time from median if not provided
    if target_time is None:
        target_time = float(model_data[duration_var].median())

    model = CoxPHFitter(penalizer=penalizer)

    return model, model_data, time_unit, formula, event_var, target_time


def create_kaplan_meier_model(
    data: pd.DataFrame,
    duration_var: str = "AVAL",
    event_var: Optional[str] = None,
    censoring_var: Optional[str] = None,
    avalu_var: str = "AVALU"
) -> Tuple[KaplanMeierFitter, pd.DataFrame, str, str]:
    """
    Configure a Kaplan-Meier model (not fitted).

    Supports CDISC ADTTE conventions.

    Args:
        data: Input DataFrame.
        duration_var: Time-to-event column (default "AVAL").
        event_var: Event indicator (1=event).
        censoring_var: Censoring indicator (1=censored).
        avalu_var: Time unit column (default "AVALU").

    Returns:
        Tuple of (model, model_data, time_unit, event_var).
    """
    if censoring_var is not None and event_var is None:
        if censoring_var not in data.columns:
            raise ValueError(
                f"Column '{censoring_var}' not found in DataFrame."
            )
        event_var = "_event_computed"
        data = data.copy()
        data[event_var] = 1 - data[censoring_var]

    for col in [duration_var, event_var]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    time_unit = "Time"
    if avalu_var and avalu_var in data.columns:
        units = data[avalu_var].dropna().unique()
        if len(units) > 0:
            time_unit = str(units[0])

    model_data = data[[duration_var, event_var]].dropna().copy()
    model = KaplanMeierFitter()

    return model, model_data, time_unit, event_var


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
    """Prepare model data for CoxPHFitter."""
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

    return data[required_cols].dropna().copy()


def _build_result_df(
    fitted_model: CoxPHFitter,
    labels: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Build Hazard Ratios DataFrame from fitted Cox model."""
    summary = fitted_model.summary.copy()

    summary['HazardRatio'] = np.exp(summary['coef'])
    summary['LowerCI']     = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['UpperCI']     = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    summary['p-value']     = summary['p'].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )

    result_df = summary[
        ['HazardRatio', 'LowerCI', 'UpperCI', 'p-value']
    ].reset_index()
    result_df = result_df.rename(
        columns={result_df.columns[0]: 'Variable'}
    )

    if labels:
        result_df['Variable'] = (
            result_df['Variable'].map(labels).fillna(result_df['Variable'])
        )

    return result_df


# ============================================================================
# UNIFIED SURVIVAL PIPELINE
# ============================================================================

class Survival(RAPID):
    """
    Unified survival analysis pipeline — Cox PH + Kaplan-Meier.

    Runs both Cox Proportional Hazards and Kaplan-Meier in a single
    pipeline. Optionally runs Log-rank test when group_var is provided.

    Cox is the primary model — fitted_model points to the fitted
    CoxPHFitter. KM is secondary — km_model stores the fitted
    KaplanMeierFitter.

    Usage:
        model = RAPID.create(data, model='survival',
                             duration_var='AVAL',
                             event_var='event',
                             independent_vars=['age', 'sex'])
        model.fit(cross_validation=True, assumptions=True)
        model.summary(plots=['forest_plot', 'survival_curve'])
    """

    def __init__(
        self,
        cox_model: CoxPHFitter,
        km_model_config: KaplanMeierFitter,
        model_data: pd.DataFrame,
        duration_var: str,
        event_var: str,
        independent_vars: List[str],
        labels: Optional[Dict[str, str]] = None,
        time_unit: str = "Time",
        formula: Optional[str] = None,
        target_time: Optional[float] = None,
        group_var: Optional[str] = None,
        **kwargs
    ):
        # Cox (primary)
        self._model         = cox_model
        # KM config (secondary)
        self._km_model_config = km_model_config

        self.model_data       = model_data
        self.duration_var     = duration_var
        self.event_var        = event_var
        self.independent_vars = independent_vars
        self.labels           = labels
        self.time_unit        = time_unit
        self.formula          = formula
        self.target_time      = target_time
        self.group_var        = group_var

        self.model_type = "survival"

        # Data matrices — X includes covariates for Cox
        covariate_cols = [
            c for c in (independent_vars or [])
            if c in model_data.columns
        ]
        self.X = (
            model_data[[duration_var] + covariate_cols]
            if covariate_cols
            else model_data[[duration_var]]
        )
        self.y = model_data[event_var]

        # Results
        self.fitted_model    = None   # Cox (primary)
        self.km_model        = None   # KM (secondary)
        self.result_df       = None   # Cox HRs
        self.km_result_df    = None   # KM summary
        self.metrics         = None
        self.plots_map       = {}
        self.logrank_results = None

        self._setup_plots_map()
        super().__init__()

    # ======================================================================
    # PLOTS MAP
    # ======================================================================

    def _setup_plots_map(self):
        """Configure available plots for unified Survival pipeline."""
        self.plots_map = {
            "forest_plot":          self._forest_plot,
            "survival_curve":       self._survival_curve,
            "schoenfeld_residuals": self._schoenfeld_residuals,
            "martingale_residuals": self._martingale_residuals,
            "deviance_residuals":   self._deviance_residuals,
            "brier_score":          self._brier_score,
            "roc_time_dependent":   self._roc_time_dependent,
            "calibration_plot":     self._calibration_plot,
        }
        if self.group_var:
            self.plots_map["compare_survival_curves"] = \
                self._compare_survival_curves

    # ======================================================================
    # CREATE
    # ======================================================================

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "survival",
        duration_var: str = "AVAL",
        event_var: Optional[str] = None,
        censoring_var: Optional[str] = None,
        independent_vars: Optional[List[str]] = None,
        formula: Optional[str] = None,
        avalu_var: str = "AVALU",
        penalizer: float = 0.1,
        target_time: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None,
        group_var: Optional[str] = None,
        **params
    ) -> "Survival":
        """
        Create unified Survival pipeline (Cox PH + Kaplan-Meier).

        Args:
            data: Input DataFrame in ARC format.
            model: Must be "survival".
            duration_var: Time-to-event column (default "AVAL").
            event_var: Event indicator (1=event). Computed from
                censoring_var if not provided.
            censoring_var: Censoring indicator (1=censored, CDISC CNSR).
            independent_vars: Covariate columns for Cox PH model.
            formula: Patsy formula for Cox. If None, built from
                independent_vars.
            avalu_var: Time unit column (CDISC ADTTE, default "AVALU").
            penalizer: Ridge penalizer for CoxPHFitter (default 0.1).
            target_time: Time point for time-dependent metrics. If None,
                uses median of duration_var.
            labels: Optional display labels for variable names.
            group_var: Optional grouping variable for KM group comparison
                and Log-rank test.

        Returns:
            Configured Survival instance ready for fit().
        """
        cox_model, model_data, time_unit, formula_used, event_var_used, \
            target_time_used = create_survival_model(
                data=data,
                duration_var=duration_var,
                event_var=event_var,
                censoring_var=censoring_var,
                independent_vars=independent_vars,
                formula=formula,
                avalu_var=avalu_var,
                penalizer=penalizer,
                target_time=target_time,
            )

        km_model_config = KaplanMeierFitter()

        return cls(
            cox_model=cox_model,
            km_model_config=km_model_config,
            model_data=model_data,
            duration_var=duration_var,
            event_var=event_var_used,
            independent_vars=independent_vars or [],
            labels=labels,
            time_unit=time_unit,
            formula=formula_used,
            target_time=target_time_used,
            group_var=group_var,
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(
        self,
        grid_search=False,
        param_grid=None,
        selection_metric="auto"
    ):
        """
        Train Cox PH (primary) and Kaplan-Meier (secondary).

        Optionally runs Log-rank test if group_var was provided.

        Returns:
            Fitted CoxPHFitter (primary fitted_model).
        """
        # ── 1. Fit Cox (primary) ──────────────────────────────────────────
        if self.formula is not None:
            cox_fitted = self._model.fit(
                self.model_data,
                duration_col=self.duration_var,
                formula=self.formula,
                event_col=self.event_var,
            )
        else:
            cox_fitted = self._model.fit(
                self.model_data,
                duration_col=self.duration_var,
                event_col=self.event_var,
            )

        # ── 2. Fit KM overall (secondary) ─────────────────────────────────
        km_fitted = KaplanMeierFitter()
        km_fitted.fit(
            self.model_data[self.duration_var],
            event_observed=self.model_data[self.event_var],
            label="Overall",
        )
        self.km_model = km_fitted

        # ── 3. KM by group (if group_var provided) ────────────────────────
        if self.group_var and self.group_var in self.model_data.columns:
            self.km_by_group = {}
            for grp, grp_df in self.model_data.groupby(self.group_var):
                km_grp = KaplanMeierFitter()
                km_grp.fit(
                    grp_df[self.duration_var],
                    event_observed=grp_df[self.event_var],
                    label=str(grp),
                )
                self.km_by_group[str(grp)] = km_grp

        # ── 4. Log-rank test (optional) ───────────────────────────────────
        if self.group_var and self.group_var in self.model_data.columns:
            try:
                from lifelines.statistics import multivariate_logrank_test
                logrank = multivariate_logrank_test(
                    self.model_data[self.duration_var],
                    self.model_data[self.group_var],
                    self.model_data[self.event_var],
                )
                self.logrank_results = {
                    'test_statistic': float(logrank.test_statistic),
                    'p_value':        float(logrank.p_value),
                    'groups': sorted(
                        self.model_data[self.group_var]
                            .dropna().unique().tolist()
                    ),
                }
            except Exception as e:
                self.logrank_results = {'error': str(e)}

        return cox_fitted

    def _build_result_df(self):
        """
        Build Cox HR DataFrame (primary result_df).

        Also populates self.km_result_df with KM summary.
        """
        # Cox HRs (primary)
        result_df = _build_result_df(self.fitted_model, self.labels)

        # KM summary (secondary)
        if self.km_model is not None:
            self.km_result_df = pd.DataFrame([
                {
                    'Variable': 'Median_Survival',
                    'Value':    self.km_model.median_survival_time_,
                },
                {
                    'Variable': 'N_Events',
                    'Value':    int(
                        self.km_model.event_table['observed'].sum()
                    ),
                },
                {
                    'Variable': 'N_Censored',
                    'Value':    int(
                        self.km_model.event_table['censored'].sum()
                    ),
                },
            ])

        return result_df

    def _calculate_metrics(self, metrics=None):
        """
        Calculate Cox and KM metrics.

        Cox (primary): C-index, AIC, partial log-likelihood, LRT.
        KM (secondary): median survival, n_events, n_censored.
        Log-rank (if group_var): test_statistic, p_value.
        """
        from isaric.modelevaluation.metrics import compute_survival_metrics

        result = compute_survival_metrics(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
        )

        # KM metrics
        if self.km_model is not None:
            result['median_survival'] = float(
                self.km_model.median_survival_time_
            )
            result['km_n_events'] = int(
                self.km_model.event_table['observed'].sum()
            )
            result['km_n_censored'] = int(
                self.km_model.event_table['censored'].sum()
            )

        # Log-rank metrics
        if (self.logrank_results
                and 'error' not in self.logrank_results):
            result['logrank_statistic'] = (
                self.logrank_results['test_statistic']
            )
            result['logrank_p_value'] = self.logrank_results['p_value']

        # Model info
        n_events = int(self.y.sum())
        self.model_info = {
            "model_type":     self.model_type,
            "framework":      "lifelines",
            "method":         "Cox PH + Kaplan-Meier",
            "random_seed":    self._random_seed,
            "N_observations": len(self.y),
            "N_events":       n_events,
            "N_censored":     len(self.y) - n_events,
            "N_features":     len(self.independent_vars),
            "time_unit":      self.time_unit,
            "parameters": {
                "duration_var":     self.duration_var,
                "event_var":        self.event_var,
                "independent_vars": self.independent_vars,
                "formula":          self.formula or "(matrix-based)",
                "penalizer":        float(self._model.penalizer)
                                    if hasattr(self._model, 'penalizer')
                                    else None,
                "target_time":      self.target_time,
                "group_var":        self.group_var,
            },
        }

        if 'c_index' in result:
            self.model_info["c_index"] = result['c_index']
        if 'lrt_statistic' in result:
            self.model_info["lrt_statistic"] = result['lrt_statistic']
        if 'lrt_p_value' in result:
            self.model_info["lrt_p_value"] = result['lrt_p_value']

        return result

    def _cross_validate(self, k_folds=5, repetitions=1):
        """
        K-fold cross-validation for Cox using Concordance Index.

        Uses formula-based fitting when formula is available, ensuring
        correct handling of categorical variables and censoring_var.
        """
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_c_indices = []

        # Columns needed for each fold
        if self.formula is not None:
            required_cols = [self.duration_var, self.event_var] + \
                            self.independent_vars
            required_cols = [
                c for c in required_cols if c in self.model_data.columns
            ]
        else:
            required_cols = (
                [self.duration_var, self.event_var] + self.independent_vars
            )

        fold_data = self.model_data[required_cols].copy()

        for train_idx, test_idx in kf.split(fold_data):
            train_df = fold_data.iloc[train_idx].reset_index(drop=True)
            test_df  = fold_data.iloc[test_idx].reset_index(drop=True)

            model_fold = CoxPHFitter(penalizer=self._model.penalizer)

            try:
                if self.formula is not None:
                    model_fold.fit(
                        train_df,
                        duration_col=self.duration_var,
                        event_col=self.event_var,
                        formula=self.formula,
                    )
                else:
                    model_fold.fit(
                        train_df,
                        duration_col=self.duration_var,
                        event_col=self.event_var,
                    )

                c_index = model_fold.score(
                    test_df,
                    scoring_method="concordance_index",
                )
                cv_c_indices.append(c_index)
            except Exception:
                pass

        return {
            'scores':     np.array(cv_c_indices),
            'mean_score': float(np.mean(cv_c_indices)),
            'std_score':  float(np.std(cv_c_indices)),
        }

    def _calibration_curve(self):
        """Survival calibration, Brier Score, and ROC."""
        import warnings
        from isaric.modelevaluation.calibration import (
            survival_calibration,
            survival_brier_score,
            survival_roc,
        )

        result = {}

        try:
            result['calibration'] = survival_calibration(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time,
            )
        except Exception as e:
            warnings.warn(f"Calibration failed: {e}")

        try:
            result['brier_score'] = survival_brier_score(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time,
            )
        except Exception as e:
            warnings.warn(f"Brier score failed: {e}")

        try:
            result['roc'] = survival_roc(
                self.fitted_model,
                self.model_data,
                duration_var=self.duration_var,
                event_var=self.event_var,
                target_time=self.target_time,
            )
        except Exception as e:
            warnings.warn(f"ROC failed: {e}")

        return result

    def _check_assumptions(self):
        """Check Proportional Hazards assumption (Schoenfeld residuals)."""
        from isaric.modelevaluation.assumptions import test_proportional_hazards
        return {
            'proportional_hazards': test_proportional_hazards(
                self.fitted_model,
                self.model_data,
            )
        }

    def _train_test_split(self, test_size=0.2):
        """Temporal holdout validation for Cox."""
        from isaric.modelevaluation.traintest import temporal_holdout
        return temporal_holdout(
            self.model_data,
            date_col=self.duration_var,
            test_size=test_size,
        )

    def _validate_external(self, external_data):
        """Validate Cox model on external independent dataset."""
        from isaric.validation.external import validate_survival
        return validate_survival(
            fitted_model=self.fitted_model,
            external_data=external_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time,
        )

    def _validate_bootstrap(self, n_iterations=1000):
        """
        Bootstrap CIs for Cox hazard ratios.

        Passes formula to bootstrap_hazard_ratios() so formula-based
        models (including those built from censoring_var) are correctly
        handled in each bootstrap iteration.

        Returns:
            Dictionary with hazard_ratios DataFrame, bootstrap_type,
            n_iterations.
        """
        from isaric.modelevaluation.bootstrap import bootstrap_hazard_ratios

        bootstrap_type = getattr(self, '_bootstrap_type', 'auto')
        if bootstrap_type == 'auto':
            bootstrap_type = 'bias_corrected'

        penalizer = (
            float(self._model.penalizer)
            if hasattr(self._model, 'penalizer') else 0.0
        )

        df = bootstrap_hazard_ratios(
            data=self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            covariates=self.independent_vars,
            n_iterations=n_iterations,
            bootstrap_type=bootstrap_type,
            random_state=getattr(self, '_random_seed', 42),
            penalizer=penalizer,
            formula=self.formula,
        )

        return {
            'hazard_ratios':  df,
            'bootstrap_type': bootstrap_type,
            'n_iterations':   n_iterations,
        }

    def _validate_sensitivity(self):
        """Sensitivity analysis."""
        try:
            from isaric.validation.sensitivity import (
                alternative_missing_handling,
            )
            return alternative_missing_handling(
                self.model_data,
                self.event_var,
                self.independent_vars,
            )
        except Exception:
            return None

    def _validate_subgroups(self, subgroups):
        """Subgroup analysis."""
        try:
            from isaric.validation.subgroup import stratified_metrics
            return stratified_metrics(
                self.fitted_model,
                self.X,
                self.y,
                subgroups,
            )
        except Exception:
            return None

    def _validate_net_benefit(self):
        """Not applicable for survival."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _forest_plot(self, backend="plotly"):
        """Forest plot of Cox hazard ratios."""
        from isaric.visualization.forestplots import hazard_ratio_plot
        return hazard_ratio_plot(
            self.result_df,
            effect_col='HazardRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot — Hazard Ratios (Cox PH)",
            backend=backend,
        )

    def _survival_curve(self, backend="plotly"):
        """Overall Kaplan-Meier survival curve."""
        from isaric.visualization.survivalcurves import kaplan_meier_curve
        return kaplan_meier_curve(
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            title="Kaplan-Meier Survival Curve",
            xaxis_title=f"Time ({self.time_unit})",
            backend=backend,
        )

    def _compare_survival_curves(self, backend="plotly"):
        """KM survival curves by group with log-rank annotation."""
        from isaric.visualization.survivalcurves import compare_survival_curves
        return compare_survival_curves(
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            group_var=self.group_var,
            logrank_results=self.logrank_results,
            title="Survival by Group",
            xaxis_title=f"Time ({self.time_unit})",
            backend=backend,
        )

    def _schoenfeld_residuals(self, backend="plotly"):
        """Schoenfeld residuals for PH assumption check."""
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
                backend=backend,
            )
            figs.append(fig)
        return figs

    def _martingale_residuals(self, backend="plotly"):
        """Martingale residuals for model covariates."""
        from isaric.visualization.survivalcurves import martingale_residuals_plot

        residuals = self.fitted_model.compute_residuals(
            self.model_data, 'martingale'
        )
        cols_to_plot = [
            c for c in self.independent_vars
            if c in self.model_data.columns
        ]
        if not cols_to_plot and self.formula:
            cols_to_plot = [
                c.strip() for c in self.formula.split('+')
                if c.strip() in self.model_data.columns
            ]

        figs = []
        for col in cols_to_plot:
            fig = martingale_residuals_plot(
                residuals=residuals['martingale'].values,
                covariate=self.model_data[col].values,
                covariate_name=col,
                add_smoother=True,
                backend=backend,
            )
            figs.append(fig)
        return figs

    def _deviance_residuals(self, backend="plotly"):
        """Deviance residuals for model covariates."""
        from isaric.visualization.survivalcurves import deviance_residuals_plot

        residuals = self.fitted_model.compute_residuals(
            self.model_data, 'deviance'
        )
        cols_to_plot = [
            c for c in self.independent_vars
            if c in self.model_data.columns
        ]
        if not cols_to_plot and self.formula:
            cols_to_plot = [
                c.strip() for c in self.formula.split('+')
                if c.strip() in self.model_data.columns
            ]

        figs = []
        for col in cols_to_plot:
            fig = deviance_residuals_plot(
                residuals=residuals['deviance'].values,
                covariate=self.model_data[col].values,
                covariate_name=col,
                backend=backend,
            )
            figs.append(fig)
        return figs

    def _brier_score(self, backend="plotly"):
        """Time-dependent Brier score plot."""
        from isaric.visualization.survivalcurves import brier_score_plot
        from isaric.modelevaluation.calibration import survival_brier_score

        data = survival_brier_score(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time,
        )
        return brier_score_plot(
            time_points=data['time_points'],
            brier_scores=data['brier_scores'],
            target_time=self.target_time,
            backend=backend,
        )

    def _roc_time_dependent(self, backend="plotly"):
        """Time-dependent ROC curve."""
        from isaric.visualization.survivalcurves import roc_time_dependent_plot
        from isaric.modelevaluation.calibration import survival_roc

        data = survival_roc(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time,
        )
        return roc_time_dependent_plot(
            fpr=data['fpr'],
            tpr=data['tpr'],
            auc=data['auc'],
            target_time=self.target_time,
            backend=backend,
        )

    def _calibration_plot(self, backend="plotly"):
        """Calibration plot for Cox at target_time."""
        from isaric.visualization.survivalcurves import calibration_plot
        from isaric.modelevaluation.calibration import survival_calibration

        data = survival_calibration(
            self.fitted_model,
            self.model_data,
            duration_var=self.duration_var,
            event_var=self.event_var,
            target_time=self.target_time,
        )
        return calibration_plot(
            predicted_survival=data['predicted_survival'],
            observed_survival=data['observed_survival'],
            target_time=self.target_time,
            backend=backend,
        )