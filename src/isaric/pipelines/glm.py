import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from isaric.pipelines.modules.rapid_plots import ResidualPlots, ForestPlot
from isaric.pipelines.regression import RAPID_BaseRegression

class RAPID_GLM(RAPID_BaseRegression):

    """
    Pipeline that enables linear regression analysis for continuous outcomes.
    This class implements linear regression as part of the ISARIC analytical pipeline,
    and generates reports useful for clinical research applied to epidemiological contexts.

    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """
    
    def __init__(self, data: pd.DataFrame, dependent_var: str = None, independent_vars: list = None, 
                formula: str = None, family: str = "gaussian", link: str = "identity", regression_type: str = "Multi"):
        super().__init__(data=data, dependent_var=dependent_var, independent_vars=independent_vars, 
                        formula=formula, family=family, link=link, regression_type=regression_type)
    
    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def fit(self, labels: dict = None, cross_val: bool = True, n_splits: int = 5):
        """
        Fits the model. 
        Calculates assumption tests, performance metrics and optionally performs cross validation.

        Args:
            labels(dict): Maps variable names to human legible names for display.
            cross_val(bool): Whether or not to perform cross validation.
            n_splits(int): Number of splits for cross validation (default: 5).

        """
        super().fit(labels, cross_val, n_splits)
        if(cross_val):
            self._evaluate_cross_validation(n_splits)

    def summary(self, assumptions=None, performance=None, cross_val=None,
                plots: list = None, vif_threshold: float = 5.0):
        """
        Reports the results of the linear regression, generating tables and plots.

        Args:
            assumptions: 'all' shows everything. A list of strings selects specific metrics.
                        Available: 'Durbin-Watson', 'Shapiro-Wilk Statistic', 'Shapiro-Wilk p-value',
                        'Influential Outliers Threshold', 'Number of Influential Points',
                        'VIF', 'Influential Outliers'. None skips.
            performance: 'all' shows everything. A list of strings selects specific metrics.
                        Available: 'MSE', 'RMSE', 'MAE', 'R2', 'Adjusted R2', 'Mcfadden R2',
                        'Adjusted Mcfadden R2', 'Efron R2', 'AIC', 'BIC', 'LLF'. None skips.
            cross_val:   'all' shows everything. A list of strings selects specific metrics.
                        Available: 'Mean CV MSE', 'Standard Deviation of CV MSE',
                        'Individual Fold MSEs'. None skips.
            plots (list): List of plots ['forest_plot', 'residuals_vs_fitted', 'qq_plot']
            vif_threshold (float): Threshold for flagging multicollinearity (default: 5.0)
        """
        self._visualization(assumptions, performance, cross_val, plots, vif_threshold)



    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    def _validation():
        pass

    def _visualization(self, assumptions=None, performance=None, cross_validation=None,
                    plots: list = None, vif_threshold: float = 5.0):
        if assumptions is not None:
            self._report_assumptions(vif_threshold, metrics=assumptions)
        if performance is not None:
            self._report_performance(metrics=performance)
        if cross_validation is not None:
            if not hasattr(self, 'cv_mse_scores') or self.cv_mse_scores is None:
                print("Cross validation not performed after fit, cannot show results.")
            else:
                self._report_cv_metrics(metrics=cross_validation)
        if plots is not None:
            if 'forest_plot' in plots:
                self._forest_plot()
            if 'residuals_vs_fitted' in plots:
                self._residuals_vs_fitted()
            if 'qq_plot' in plots:
                self._qq_plot()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTION TESTS EVALUATION)
    # ------------------------------------------------------------------

    def _evaluate_independence_of_errors(self):
        self.dw = self.assumption_tester.test_durbin_watson()
    
    def _evaluate_normality_of_errors_shapiro_wilk(self):
        shapiro_test_results = self.assumption_tester.test_normality()
        self.shapiro_wilk_test_statistic = shapiro_test_results["statistic"]
        self.shapiro_wilk_p_value = shapiro_test_results["p_value"]

    def _build_assumption_metrics_df(self, vif_threshold: float = 5.0):
        """
        Build a dataframe containing all assumption test metrics.
        """
        # Durbin-Watson interpretation
        if self.dw < 1.5:
            dw_interpretation = "Positive autocorrelation"
        elif self.dw > 2.5:
            dw_interpretation = "Negative autocorrelation"
        else:
            dw_interpretation = "Residuals likely independent"
        
        # Shapiro-Wilk interpretation
        if self.shapiro_wilk_p_value > 0.05:
            shapiro_interpretation = "Normally distributed (fail to reject H0)"
        else:
            shapiro_interpretation = "Not normally distributed (reject H0)"
        
        # Truncate influential points to first 5
        influential_points_display = self.influential_points[:5]
        if len(self.influential_points) > 5:
            influential_points_str = f"{influential_points_display} ... ({len(self.influential_points)} total)"
        else:
            influential_points_str = str(self.influential_points)
        
        # Build the dataframe
        assumption_data = {
            'Test': [
                'Durbin-Watson',
                'Shapiro-Wilk Statistic',
                'Shapiro-Wilk p-value',
                'Influential Outliers Threshold',
                'Number of Influential Points'
            ],
            'Value': [
                f"{self.dw:.3f}",
                f"{self.shapiro_wilk_test_statistic:.4f}",
                f"{self.shapiro_wilk_p_value:.4f}",
                f"{self.influential_outliers_threshold:.3f}",
                len(self.influential_points)
            ],
            'Interpretation': [
                dw_interpretation,
                shapiro_interpretation,
                '',
                f"Points above threshold: {influential_points_str}",
                ''
            ]
        }
        
        self.assumption_metrics_df = pd.DataFrame(assumption_data)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS EVALUATION)
    # ------------------------------------------------------------------

    def _evaluate_mse(self):
        self.mse = mean_squared_error(self.y, self.fitted_model.fittedvalues)
        self.rmse = np.sqrt(self.mse)

    def _evaluate_mae(self):
        self.mae = mean_absolute_error(self.y, self.fitted_model.fittedvalues)

    def _evaluate_r2(self):
        p = int(self.fitted_model.df_model)
        n = int(self.fitted_model.nobs)
        self.r2 = r2_score(self.y, self.fitted_model.fittedvalues)
        self.adjusted_r2 = 1 - (1 - self.r2) * ((n - 1) / (n - p - 1))

    def _evaluate_cross_validation(self, n_splits):
        """
        Perform k-fold cross-validation using the same statsmodels GLM.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_mse_scores = []
        
        for train_idx, test_idx in kf.split(self.X):
            # Split data
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            
            # Fit statsmodels GLM on training fold
            model_fold = sm.GLM(endog=y_train, exog=X_train, family=self.family)
            result_fold = model_fold.fit()
            
            # Predict on test fold
            y_pred = result_fold.predict(X_test)
            
            # Calculate MSE for this fold
            mse = mean_squared_error(y_test, y_pred)
            cv_mse_scores.append(mse)
        
        self.cv_mse_scores = np.array(cv_mse_scores)

    def _build_performance_metrics_df(self):
        def fmt(v):
            return 'N/A' if (isinstance(v, float) and np.isnan(v)) else f'{v:.6f}'

        performance_data = {
            'Metric': ['MSE','RMSE','MAE','R2','Adjusted R2',
                    'Mcfadden R2','Adjusted Mcfadden R2','Efron R2','AIC','BIC','LLF'],
            'Value':  [fmt(self.mse), fmt(self.rmse), fmt(self.mae),
                    fmt(self.r2), fmt(self.adjusted_r2),
                    fmt(self.mcfadden_r2), fmt(self.mcfadden_adj_r2), fmt(self.efron_r2),
                    fmt(self.aic), fmt(self.bic), fmt(self.llf)]
        }
        self.performance_metrics_df = pd.DataFrame(performance_data)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_performance(self, metrics=None):
        if self.performance_metrics_df is None:
            self._build_performance_metrics_df()

        df = self.performance_metrics_df
        if metrics != 'all':
            missing = set(metrics) - set(df['Metric'])
            if missing:
                print(f"Warning: the following performance metrics were not found: {missing}")
            df = df[df['Metric'].isin(metrics)]

        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (CROSS VALIDATION METRICS VISUALIZATION)
    # ------------------------------------------------------------------

    def _build_cv_df(self):
        cv_data = {
            'Metric': [
                'Mean CV MSE',
                'Standard Deviation of CV MSE',
                'Individual Fold MSEs'
            ],
            'Value': [
                f"{np.mean(self.cv_mse_scores):.6f}",
                f"{np.std(self.cv_mse_scores):.6f}",
                ', '.join([f"{score:.6f}" for score in self.cv_mse_scores])
            ]
        }
        self.cv_df = pd.DataFrame(cv_data)

    def _report_cv_metrics(self, metrics=None):
        cv_df = self.cv_df
        if metrics != 'all':
            missing = set(metrics) - set(cv_df['Metric'])
            if missing:
                print(f"Warning: the following CV metrics were not found: {missing}")
            cv_df = cv_df[cv_df['Metric'].isin(metrics)]
        print("=" * 80)
        print("CROSS-VALIDATION METRICS")
        print("=" * 80)
        print(cv_df.to_string(index=False))
        print("=" * 80)
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR FOR FIT)
    # ------------------------------------------------------------------
    def _rename_cols_by_regression_type(self):
        config = self._link_display_config
        label  = config['effect_label']
        suffix = '(uni)' if self.regression_type.lower() == 'uni' else '(multi)'

        self.summary_df.rename(columns={
            label:       f'{label} {suffix}',
            'Lower CI':  f'Lower CI {suffix}',
            'Upper CI':  f'Upper CI {suffix}',
            'p-value':   f'p-value {suffix}'
        }, inplace=True)

    def _build_result_summary_df(self, labels: dict = None):
        summary_table = self.summary_table
        p_value_col = 'P>|z|' if 'P>|z|' in summary_table.columns else 'P>|t|'

        self.summary_df = summary_table[[
            'Coef.', '[0.025', '0.975]', p_value_col
        ]].reset_index()

        self.summary_df = self.summary_df.rename(columns={
            'index':     'Variable',
            'Coef.':     'Coefficient',
            '[0.025':    'LowerCI',
            '0.975]':    'UpperCI',
            p_value_col: 'p-value'
        })

        config = self._link_display_config
        if config['exp_coef']:
            self.summary_df[['Coefficient', 'LowerCI', 'UpperCI']] = \
                np.exp(self.summary_df[['Coefficient', 'LowerCI', 'UpperCI']])

        # Rename columns to reflect actual content
        label = config['effect_label']
        self.summary_df = self.summary_df.rename(columns={
            'Coefficient': label,
            'LowerCI':     f'Lower CI',
            'UpperCI':     f'Upper CI',
        })

        self.summary_df = self._map_variable_label(self.summary_df, labels)
        self.summary_df = self.summary_df[['Variable', label, 'Lower CI', 'Upper CI', 'p-value']]

    # ------------------------------------------------------------------
    # PLOT GENERATION AND REPORTING
    # ------------------------------------------------------------------

    def _residuals_vs_fitted(self):
        """Generate and display residuals vs fitted values plot."""
        fig = ResidualPlots.residuals_vs_fitted(
            residuals=self.fitted_model.resid_response,
            fitted_values=self.fitted_model.fittedvalues,
            title="Residuals vs Adjusted Values",
            xlabel="Adjusted Values",
            ylabel="Residuals"
        )
        fig.show()
    
    def _qq_plot(self):
        """Generate and display Q-Q plot for normality assessment."""
        fig = ResidualPlots.qq_plot(
            residuals=self.fitted_model.resid_response,
            title='Normality of Errors: Q-Q Plot'
        )
        fig.show()
    
    def _forest_plot(self):
        df = self.summary_df.copy()
        # Strip uni/multi suffixes
        df.columns = df.columns.str.replace(r' \((uni|multi)\)', '', regex=True)
        # Also strip bare suffixes added by _rename_cols_by_regression_type
        df.columns = df.columns.str.replace(r' \(uni\)| \(multi\)', '', regex=True)

        config = self._link_display_config
        label  = config['effect_label']

        # Resolve actual column names after stripping
        effect_col = label
        lower_col  = 'Lower CI'
        upper_col  = 'Upper CI'

        fig = ForestPlot.plot(
            df=df,
            label_col='Variable',
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title="Forest Plot" + (f" ({self.regression_type})" if hasattr(self, 'regression_type') else ""),
            xaxis_title=label,
            null_value=config['null_value'],
            log_scale=config['log_scale']
        )
        fig.show()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (MODEL EVALUATION)
    # ------------------------------------------------------------------

    def _test_performance_metrics(self):
        super()._test_performance_metrics()
        self._evaluate_mse()
        self._evaluate_mae()
        self._evaluate_r2()
        self._build_performance_metrics_df()

    def _test_assumptions(self):
        self._setup_assumption_tester()
        self._evaluate_independence_of_errors()
        self._evaluate_vif()
        self._evaluate_normality_of_errors_shapiro_wilk()
        self._evaluate_influential_outliers()
        self._build_assumption_metrics_df()
    
    def _test_cross_validation(self, n_splits):
        self._evaluate_cross_validation(n_splits)
        self._build_cv_df()

    # ------------------------------------------------------------------
    # PROPERTIES FOR THIS REGRESSION 
    # ------------------------------------------------------------------
    @property
    def _family_map(self):
        return {
            "gaussian":     sm.families.Gaussian,
            "gamma":        sm.families.Gamma,
            "inv_gaussian": sm.families.InverseGaussian,
            "tweedie":      sm.families.Tweedie,
            }

    @property
    def _link_map(self):
        return {
            "identity": sm.families.links.Identity,
            "log":      sm.families.links.Log,
            "inverse":  sm.families.links.InversePower,
            "sqrt":     sm.families.links.Sqrt,
            }
    
    # ------------------------------------------------------------------
    # LINK DISPLAY CONFIGURATION
    # ------------------------------------------------------------------
    @property
    def _link_display_config(self):
        is_log     = isinstance(self.fitted_model.family.link, sm.families.links.Log)
        is_poisson = isinstance(self.fitted_model.family, sm.families.Poisson)
        is_tweedie = isinstance(self.fitted_model.family, sm.families.Tweedie)
        is_gamma   = isinstance(self.fitted_model.family, sm.families.Gamma)
        is_inv_gau = isinstance(self.fitted_model.family, sm.families.InverseGaussian)

        if is_log and is_poisson:
            effect_label = 'Risk Ratio'
        elif is_log:
            effect_label = 'Mean Ratio'
        else:
            effect_label = 'Coefficient'

        return {
            'exp_coef':     is_log,
            'null_value':   1.0 if is_log else 0.0,
            'log_scale':    is_log,
            'effect_label': effect_label,
        }