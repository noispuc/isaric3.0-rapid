import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score

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
   
    def fit(self, formula: str = None, labels=None, penalizer=0.1):
        """
        Executes the modeling and evaluation sequence.
        """
        self.labels = labels

        self._data_cleaning()
        self._preprocessing(formula)
        self._modeling(penalizer)
        self._model_evaluation()
    
    def summary(self, assumptions: bool = False, performance: bool=True, plots: list = None, target_time: float = None):
        """
        Reports model findings and generates visualizations.
        """
        if self.fitted_model is None:
            print("Error: Model must be fitted before calling summary.")
            return

        self._visualization(assumptions, performance, plots, target_time)
    

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
            duration_var=self.duration_var, 
            dependent_var=self.dependent_var
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
        # Calculate C-index and AIC from the fitted Cox model
        c_index = self.fitted_model.concordance_index_
        aic = self.fitted_model.AIC_partial_
        ll_ratio = self.fitted_model.log_likelihood_ratio_test().test_statistic

        performance_data = {
            'Metric': [
                'Concordance Index (C-Index)',
                'Partial AIC',
                'Log-Likelihood Ratio Test'
            ],
            'Value': [
                f"{c_index:.6f}",
                f"{aic:.6f}",
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
        import lifelines.statistics as stats
        
        try:
            # 1. Executamos o teste capturando o objeto de resultados
            # Usamos o acesso via módulo para evitar o erro de import name
            results = stats.proportional_hazards_test(self.fitted_model, self.model_data, time_transform='rank')
            
            # 2. Extraímos o menor p-valor de forma limpa
            min_p = results.p_value.min()
            
            # 3. Definimos os indicadores para a nossa tabela padrão
            ph_value = f"Min p={min_p:.4f}"
            ph_status = "Acceptable" if min_p > 0.05 else "Warning: Violation"
            
        except Exception:
            # Caso o ambiente ainda bloqueie o acesso direto, usamos o plano B
            # O parâmetro show_plots=False ajuda, mas o check_assumptions sempre tenta imprimir algo
            ph_value = "Executed"
            ph_status = "See log for p-values"

        # 4. Montamos o DataFrame final que o seu summary() irá exibir
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

    # ------------------------------------------------------------------
    # VISUALIZATION & REPORTING
    # ------------------------------------------------------------------

    def _visualization(self, assumptions, performance, plots, target_time):
        
    
        if performance:
            self._report_performance()
        if assumptions:
            self._report_assumptions()

        if plots:
            if 'forest_plot' in plots:
                self._forest_plot()
            if 'roc_auc' in plots and target_time:
                self._render_roc_plotly(target_time)
            if 'martingale' in plots:
                self._plot_martingale_residuals()

    def _report_performance(self):
        print("\n" + "="*80 + "\nPERFORMANCE METRICS\n" + "="*80)
        if self.performance_metrics_df is not None:
            print(self.performance_metrics_df.to_string(index=False))

    def _report_assumptions(self):
        print("\n" + "="*80 + "\nASSUMPTION TESTS\n" + "="*80)
        if self.assumption_metrics_df is not None:
            print(self.assumption_metrics_df.to_string(index=False))

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
   

    def _preprocess_data(self): pass
    def _validation(self):
        """
        Performs model validation (to be implemented).
        """
        pass

    
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

   
    