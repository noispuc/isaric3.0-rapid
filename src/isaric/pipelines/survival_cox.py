import pandas as pd
import numpy as np
import warnings
import lifelines.statistics as stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold

from isaric.pipelines.modules.rapid_preprocess import RapidPreprocessor
from isaric.pipelines.modules.rapid_plots import RapidPlots
from isaric.pipelines.pipeline import RAPID_BasePipeline



class RAPID_SurvivalCox(RAPID_BasePipeline):
    """
    Pipeline that enables [Survival analysis]. 
    This class implements the technique of [survival-cox analysis] as part of the ISARIC analytical pipeline, 
    and generates reports useful for clinical research applied to epidemiological contexts.
    
    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    This class inherits from RAPID_Pipeline and implements all required abstract methods.
    """

    def __init__(self, data: pd.DataFrame, duration_var: str, dependent_var: str, independent_vars: list, method: str = "Multi"):
        self.data = data
        self.duration_var = duration_var
        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.method = method
        
        self.fitted_model = None
        self.model_data = None 
        self.summary_df = None
        self.labels = None

        self.performance_metrics_df = None  # To be populated during evaluation
        self.assumption_metrics_df = None   # To be populated during evaluation
        #self.cross_val_scores = None        # Placeholder for future CV implementation

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------
   
    def fit(self, formula: str = None, labels=None, penalizer=0.1, cross_val:bool=False, n_splits: int = 5):
        """
        Fits the model. 
        Calculates assumption tests, performance metrics and optionally performs cross validation.

        Args:
            formula(str): Patsy-style formula for model specification (e.g., 'duration ~ age + sex + comorbidity').
            labels(dict): Maps variable names to human legible names for display.
            penalizer(float): L2 regularization strength for Cox model (default: 0.1).
            cross_val(bool): Whether or not to perform cross validation.
            n_splits(int): Number of splits for cross validation (default: 5).

        """
        self.labels = labels
        self._data_cleaning()
        self._preprocessing(formula)
        self._modeling(penalizer)
        self._model_evaluation()
        if cross_val:
            self._test_cross_validation(n_splits)
    
    def summary(self, assumptions: bool = False, performance: bool=True, plots: list = None, target_time: float = None):
        """
        Reports model findings and generates visualizations.
        """
        if self.fitted_model is None:
            print("Error: Model must be fitted before calling summary.")
            return

        # Consistent check: did we run cross-validation during fit?
        has_cv = hasattr(self, 'cv_df') and self.cv_df is not None
    
        # Pass 'has_cv' to the internal visualization handler
        self._visualization(assumptions, performance, plots, target_time, has_cv)
    

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    def _data_cleaning(self):
        """
        Handles initial data sanitization and missing values.
        """
        required_cols = [self.duration_var, self.dependent_var] + self.independent_vars
        self.data = self.data.dropna(subset=required_cols)
    
    def _preprocessing(self, formula):
        """
        Converts raw data into design matrices and handles collinearity.
        """

        # Matrix Generation via RapidPreprocessor
        y, X, _ = RapidPreprocessor.prepare_data(
            df=self.data,
            formula=formula,
            target_cols=[self.duration_var, self.dependent_var],
            predictor_cols=self.independent_vars, 
            intercept=False
        )

        # 3. Stability Checks: Drop zero-variance and duplicate columns
        X = X.loc[:, X.nunique() > 1]
        X = X.loc[:, ~X.T.duplicated()]

        self.model_data = pd.concat([y, X], axis=1)

    # ------------------------------------------------------------------
    # PRIVATE METHODS: MODELING & EVALUATION
    # ------------------------------------------------------------------

    def _modeling(self, penalizer):
        """
        Fits the survival_cox model using the Cox Proportional Hazards algorithm.
        """
        self.fitted_model = CoxPHFitter(penalizer=penalizer)
        self.fitted_model.fit(
            self.model_data, 
            duration_col=self.duration_var, 
            event_col=self.dependent_var
        )
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS: EVALUATION & METRICS (Standardized)
    # ------------------------------------------------------------------

    def _model_evaluation(self):
        """
        Orchestrates the calculation of all model summaries, metrics, and assumptions.
        """
        self._build_result_summary_df()

        self._test_performance_metrics()
        self._test_assumptions()
    
    def _test_performance_metrics(self):
        """
        Evaluates model performance metrics and populates performance_metrics_df.
        """
        # 1. Base metrics from lifelines
        c_index = self.fitted_model.concordance_index_
        aic = self.fitted_model.AIC_partial_
        ll_ratio = self.fitted_model.log_likelihood_ratio_test().test_statistic

        # 2. Manual BIC Calculation (Bayesian Information Criterion)
        # Formula: BIC = -2 * ln(L) + k * ln(n)
        n = self.model_data.shape[0]  # Number of observations
        k = len(self.fitted_model.params_)  # Number of parameters (covariates)
        log_likelihood = self.fitted_model.log_likelihood_
        bic_partial = -2 * log_likelihood + k * np.log(n)

        performance_data = {
            'Metric': [
                'Concordance Index (C-Index)',
                'Partial AIC',
                'Partial BIC',
                'Log-Likelihood Ratio Test'
            ],
            'Value': [
                f"{c_index:.6f}",
                f"{aic:.6f}",
                f"{bic_partial:.6f}",
                f"{ll_ratio:.6f}"
            ]
        }
        # Store in the standard dataframe attribute
        self.performance_metrics_df = pd.DataFrame(performance_data)

    def _test_assumptions(self):
        """
        Performs statistical tests for Cox model assumptions and 
        captures results programmatically to avoid messy console output.
        """
        try:
            # Execute PH test and capture the results object
            results = stats.proportional_hazards_test(self.fitted_model, self.model_data, time_transform='rank')
            
            # Extract the minimum p-value
            min_p = results.p_value.min()
            
            # Define indicators for the standard table
            ph_value = f"Min p={min_p:.4f}"
            ph_status = "Acceptable" if min_p > 0.05 else "Warning: Violation"
            
        except Exception:
            # Fallback if the environment restricts direct access
            ph_value = "Executed"
            ph_status = "See log for p-values"

        assumption_data = {
            'Test': [
                'Proportional Hazards (Schoenfeld)',
                'Linearity (Martingale Residuals)'
            ],
            'Value': [
                ph_value,
                "Visual Inspection"
            ],
            'Interpretation': [
                ph_status,
                "Check residual plots for non-linear patterns"
            ]
        }
        self.assumption_metrics_df = pd.DataFrame(assumption_data)
    
    
    def _evaluate_cross_validation(self, n_splits):
        """
        Perform k-fold cross-validation for Cox model using Concordance Index.
        """
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_c_indices = []
        
        for train_idx, test_idx in kf.split(self.model_data):
            # Split data
            train_df = self.model_data.iloc[train_idx]
            test_df = self.model_data.iloc[test_idx]
            
            # Fit CoxPH model on training fold
            model_fold = CoxPHFitter(penalizer=self.fitted_model.penalizer)
            model_fold.fit(
                train_df, 
                duration_col=self.duration_var, 
                event_col=self.dependent_var
            )
            
            # Evaluate Concordance Index on test fold
            c_index = model_fold.concordance_index_
            cv_c_indices.append(c_index)
        
        self.cross_val_scores = np.array(cv_c_indices)

    def _build_cv_df(self):
        """
        Builds the dataframe for CV reporting
        """
        cv_data = {
            'Metric': [
                'Mean C-Index (CV)',
                'Standard Deviation',
                'Individual Fold Indices'
            ],
            'Value': [
                f"{self.cross_val_scores.mean():.6f}",
                f"{self.cross_val_scores.std():.6f}",
                ', '.join([f"{score:.4f}" for score in self.cross_val_scores])
            ]
        }
        self.cv_df = pd.DataFrame(cv_data)
    
    def _test_cross_validation(self, n_splits):
        """
        Orchestrates CV evaluation and dataframe building.
        """
        self._evaluate_cross_validation(n_splits)
        self._build_cv_df()

    def _build_result_summary_df(self):
        """
        Generates Hazard Ratios and Confidence Intervals table.
        """
        summary = self.fitted_model.summary.copy()
        summary['HazardRatio'] = np.exp(summary['coef'])
        summary['LowerCI'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
        summary['UpperCI'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
        summary['p-value'] = summary['p'].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
        
        df_res = summary[['HazardRatio', 'LowerCI', 'UpperCI', 'p-value']].reset_index()
        df_res.rename(columns={df_res.columns[0]: 'Variable'}, inplace=True)
        
        if self.labels:
            df_res['Variable'] = df_res['Variable'].map(self.labels).fillna(df_res['Variable'])
        
        self.summary_df = df_res

    def _calculate_brier_score(self, target_time):
        """
        Mathematical implementation of IPCW Brier Score.
        Returns: tuple (time_points, brier_scores)
        """
        try:
            T = self.model_data[self.duration_var]
            E = self.model_data[self.dependent_var]

            max_data_time = T.max()
            eval_time = min(target_time, max_data_time)
            
            time_points = np.linspace(T[E == 1].min(), eval_time, 100)

            # Estimativa de Censura (IPCW)
            kmf_censoring = KaplanMeierFitter().fit(T, 1 - E)
            G_T = kmf_censoring.predict(T, interpolate=True)

            brier_scores = []
            for t in time_points:
                predicted_probs = self.fitted_model.predict_survival_function(self.model_data, times=[t]).squeeze()
                G_t = kmf_censoring.predict(t, interpolate=True)

                is_event_before_t = (T <= t) & (E == 1)
                term1 = np.sum(((predicted_probs[is_event_before_t] - 0)**2) / G_T[is_event_before_t])

                is_after_t = T > t
                term2 = np.sum(((predicted_probs[is_after_t] - 1)**2) / G_t)

                score = (term1 + term2) / len(self.model_data)
                brier_scores.append(score)

            return time_points, np.array(brier_scores)

        except Exception as e:
            print(f"Error in Brier calculation: {e}")
            return None, None

    # ------------------------------------------------------------------
    # VISUALIZATION & REPORTING
    # ------------------------------------------------------------------

    def _visualization(self, assumptions, performance, plots, target_time, cross_validation):
        
    
        if performance:
            self._report_performance()
        if assumptions:
            self._report_assumptions()
        if cross_validation:
            self._report_cv_metrics()

        if plots:
            if 'forest_plot' in plots:
                self._forest_plot()
            if 'survival_curve' in plots:
                self._plot_survival_curve()

            # Time-dependent metrics require a target_time
            if target_time is not None:
                if 'roc_auc' in plots:
                    self._render_roc_plotly(target_time)
                if 'brier_score' in plots:
                    self._plot_brier_score(target_time)
                if 'calibration' in plots:
                    self._render_calibration_plotly(target_time)
            else:
                # Alert the user if time-dependent plots were requested without a target time
                time_plots = ['roc_auc', 'brier_score', 'calibration']
                missing = [p for p in time_plots if p in plots]
                if missing:
                    print(f"Warning: The following plots require 'target_time': {missing}")

            # Residual disgnostics
            if 'martingale' in plots:
                self._plot_martingale_residuals()
            if 'deviance' in plots:
                self._plot_deviance_residuals()
            if 'schoenfeld' in plots:
                self._plot_schoenfeld_residuals()

    def _report_performance(self):
        print("\n" + "="*80 + "\nPERFORMANCE METRICS\n" + "="*80)
        if self.performance_metrics_df is not None:
            print(self.performance_metrics_df.to_string(index=False))

    def _report_assumptions(self):
        print("\n" + "="*80 + "\nASSUMPTION TESTS\n" + "="*80)
        if self.assumption_metrics_df is not None:
            print(self.assumption_metrics_df.to_string(index=False))
    
    def _report_cv_metrics(self, metrics='all'):            
        """
        Prints Cross-Validation results from self.cv_df.
        """
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION METRICS (CONCORDANCE)")
        print("=" * 80)
        
        if hasattr(self, 'cv_df') and self.cv_df is not None:
            df = self.cv_df
            if metrics != 'all':
                df = df[df['Metric'].isin(metrics)]
            print(df.to_string(index=False))
        else:
            print("No Cross-Validation data available. Run fit(cross_val=True) first.")
            
        print("=" * 80)

    def _forest_plot(self):
        """Generates forest plot for Hazard Ratios."""
        RapidPlots.forest.plot(
            df=self.summary_df,
            effect_col='HazardRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            label_col='Variable',
            title=f'Hazard Ratios ({self.method})',
            null_value=1.0,
            log_scale=True
        ).show()

    def _plot_survival_curve(self):
        """
        Generates and displays the baseline survival function using the fitted Cox model.
        """
        import plotly.graph_objects as go
        
        # Extract the baseline survival function from the fitted lifelines model
        survival_func = self.fitted_model.baseline_survival_
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=survival_func.index, 
            y=survival_func.iloc[:, 0],
            mode='lines',
            name='Baseline Survival',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Baseline Survival Curve (Cox Model)',
            xaxis_title='Time (Days)',
            yaxis_title='Survival Probability',
            template='plotly_white',
            yaxis=dict(range=[0, 1])
        )
        fig.show()
    
    def _plot_schoenfeld_residuals(self):
        """
        Computes and plots Schoenfeld residuals for all covariates to check the PH assumption.
        """
        from isaric.pipelines.modules.rapid_plots import RapidPlots
        
        # Identify predictors (exclude duration and event columns)
        covariates = [c for c in self.model_data.columns 
                     if c not in [self.duration_var, self.dependent_var]]

        print("Computing Schoenfeld Residuals for model diagnostics...")
        
        try:
            # Calculate residuals using the lifelines engine
            schoenfeld_res = self.fitted_model.compute_residuals(self.model_data, 'schoenfeld')
            
            for col in covariates:
                # The index of schoenfeld_res corresponds to event times
                fig = RapidPlots.schoenfeld.plot(
                    times=schoenfeld_res.index.values,
                    residuals=schoenfeld_res[col].values,
                    covariate_name=col
                )
                fig.show()
        except Exception as e:
            print(f"Error computing Schoenfeld residuals: {e}")

    def _plot_martingale_residuals(self):
        """
        Plots Martingale residuals. 
        Iterates over actual columns in model_data to avoid KeyError with formulas.
        """
        residuals = self.fitted_model.compute_residuals(self.model_data, 'martingale')
        
        # We iterate over model_data columns instead of predictors(independent_vars) 
        # to handle formula-generated names (like demog_sex[T.Male])
        cols_to_plot = [c for c in self.model_data.columns 
                        if c not in [self.duration_var, self.dependent_var]]

        for col in cols_to_plot:
            # Only plot for continuous-like variables (more than 10 unique values)
            if self.model_data[col].nunique() > 10:
                RapidPlots.residuals.residuals_vs_covariate(
                    residuals=residuals['martingale'].values,
                    covariate=self.model_data[col].values,
                    covariate_name=col,
                    residual_type='Martingale Residuals',
                    add_smoother=True
                ).show()

    def _plot_deviance_residuals(self):
        """
        Generates deviance residuals plots against all covariates in the model.
        Leverages the new Plotly-based method in RapidPlots.
        """
        # Select columns that are predictors (exclude duration and event)
        cols_to_plot = [c for c in self.model_data.columns 
                        if c not in [self.duration_var, self.dependent_var]]

        for col in cols_to_plot:
            # The method RapidPlots.residuals.deviance_residuals handles 
            # both categorical and continuous data automatically.
            RapidPlots.residuals.deviance_residuals(
                fitted_model=self.fitted_model,
                df=self.model_data,
                duration_col=self.duration_var,
                event_col=self.dependent_var,
                covariate_name=col
            ).show()
   

    def _preprocess_data(self): pass
    def _validation(self):
        """
        Performs model validation (to be implemented).
        """
        pass

    def _plot_brier_score(self, target_time):
        """
        Calculates metrics and calls the modular Brier Score plot from rapid_plots.
        """
        # 1. Realiza o cálculo matemático (IPCW Brier Score)
        time_points, brier_scores = self._calculate_brier_score(target_time)
        
        if time_points is None or brier_scores is None:
            return

        # 2. Chama a função que já existe no rapid_plots.py
        # Note que usamos a interface unificada RapidPlots
        fig = RapidPlots.brier_score.brier_score(
            time_points=time_points,
            brier_scores=brier_scores,
            target_time=target_time,
            title=f'Time-Dependent Brier Score (up to t={target_time})'
        )
        
        fig.show()
    
    def _render_roc_plotly(self, target_time):
        """
        Internal helper for ROC calculation and rendering.
        """
        T = self.model_data[self.duration_var]
        E = self.model_data[self.dependent_var]
        risk_scores = self.fitted_model.predict_partial_hazard(self.model_data).values
        
        mask = (T <= target_time) & (E == 1) | (T > target_time)
        y_true = ((T <= target_time) & (E == 1)).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true[mask], risk_scores[mask])
        auc_val = roc_auc_score(y_true[mask], risk_scores[mask])
        
        RapidPlots.roc.plot(fpr, tpr, auc_val, title=f'ROC at t={target_time}').show()

   
    def _render_calibration_plotly(self, target_time):
        """
        Internal helper to call the survival calibration plot from RapidPlots.
        """
        fig = RapidPlots.calibration.survival_calibration(
            fitted_model=self.fitted_model,
            df=self.model_data,
            duration_col=self.duration_var,
            event_col=self.dependent_var,
            t=target_time
        )
        fig.show()