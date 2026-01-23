import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score

from statsmodels.stats.stattools import durbin_watson

from isaric.pipelines.modules.rapid_plots import ResidualPlots, ForestPlot
from .regression import RAPID_BaseRegression
from isaric.pipelines.modules.rapid_assumption import ModelAssumptionTester

class RAPID_LinearRegression(RAPID_BaseRegression):

    """
    Pipeline that enables linear regression analysis for continuous outcomes.
    This class implements linear regression as part of the ISARIC analytical pipeline,
    and generates reports useful for clinical research applied to epidemiological contexts.

    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """
    
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        super().__init__(data=data, outcome_str=outcome_str, predictors_list=predictors_list, regression_type=regression_type)
    
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

    def summary(self, assumptions: bool = False, performance: bool = False, cross_val: bool = False,
                plots: list = None, vif_threshold: float = 5.0):
        """
        Reports the results of the linear regression, generating tables and plots.

        Args:
            assumptions (bool): If True, shows results of assumptions checks (Independence of Errors, Normality of Errors, Multicolinearity, Influential Outliers)
            performance (bool): If True, shows performance metrics (MSE, RMSE, MAE, R^2, Adjusted R^2)
            plots (list, optional): List of plots ['forest_plot', 'residuals_vs_fitted', 'qq_plot']
            cross_val (bool): Whether or not to show cross validation results.
            vif_threshold (float): Threshold for flagging multicollinearity (default: 5.0)
        """
        self._visualization(assumptions,performance,cross_val, plots, vif_threshold)



    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    def _validation():
        pass

    def _visualization(self, assumptions: bool = True, performance: bool = False, cross_validation: bool = False, 
                       plots: list = None, vif_threshold: float = 5.0):
        if (assumptions):
            self._report_independence_of_errors()
            self._report_normality_of_errors_shapiro()
            self._report_vif(vif_threshold)
            self._report_influential_outliers()
        if (performance):
            self._report_mse()
            self._report_rmse()
            self._report_mae()
            self._report_r2()
            self._report_adjusted_r2()
        if (cross_validation):
            if not hasattr(self, 'cv_mse_scores') or self.cv_mse_scores is None:
                print("Cross validation not performed after fit, cannot show results.")
            else:
                self._report_cv_mse()
                self._report_cv_mean_mse()
                self._report_cv_sd_mse()
        if plots is not None:
            if('forest_plot' in plots):
                self._report_forest_plot()
            if('residuals_vs_fitted' in plots):
                self._report_residuals_vs_fitted_plot()
            if('qq_plot' in plots):
                self._report_qq_plot()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTION TESTS EVALUATION)
    # ------------------------------------------------------------------

    def _evaluate_independence_of_errors(self):
        self.dw = self.assumption_tester.test_durbin_watson()
    
    def _evaluate_normality_of_errors_shapiro_wilk(self):
        shapiro_test_results = self.assumption_tester.test_normality()
        self.shapiro_wilk_test_statistic = shapiro_test_results["statistic"]
        self.shapiro_wilk_p_value = shapiro_test_results["p_value"]
    
    def _evaluate_influential_outliers(self):
        #In the rapid assumption module, there is a model agnostic computation for cook's distance
        #However, this is not used in this computation because the statsmodels implementation falls back on C and is much faster.
        influence = self.model.get_influence()
        self.cooks_d = influence.cooks_distance[0]

        threshold = 4 / len(self.cooks_d)
        self.influential_outliers_threshold = threshold

        self.influential_points = [i for i, val in enumerate(self.cooks_d) if val > threshold]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTION TESTS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_independence_of_errors(self):
        print(f'Durbin-Watson Statistic: {self.dw:.3f}')
        if self.dw < 1.5:
            print("→ Indicates positive autocorrelation of residuals.")
        elif self.dw > 2.5:
            print("→ Indicates negative autocorrelation of residuals.")
        else:
            print("→ Residuals are likely independent.")

    def _report_normality_of_errors_shapiro(self):
        print("=" * 60)
        print(f"Shapiro–Wilk test statistic: {self.shapiro_wilk_test_statistic:.4f}")
        print(f"Shapiro–Wilk p-value: {self.shapiro_wilk_p_value:.4f}")

        if self.shapiro_wilk_p_value > 0.05:
            print("Residuals appear to be normally distributed (fail to reject H0).")
        else:
            print("Residuals do not appear to be normally distributed (reject H0).")

    def _report_influential_outliers(self):
        print(f"Above limit points ({self.influential_outliers_threshold:.3f}): {self.influential_points}")
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS EVALUATION)
    # ------------------------------------------------------------------

    def _evaluate_mse(self):
        self.mse = mean_squared_error(self.y, self.model.fittedvalues)
        self.rmse = np.sqrt(self.mse)

    def _evaluate_mae(self):
        self.mae = mean_absolute_error(self.y, self.model.fittedvalues)

    def _evaluate_r2(self):
        p = int(self.model.df_model)
        n = int(self.model.nobs)
        self.r2 = r2_score(self.y, self.model.fittedvalues)
        self.adjusted_r2 = 1 - (1 - self.r2) * ((n - 1) / (n - p - 1))

    def _evaluate_cross_validation(self, n_splits):
        model_cv = LinearRegression()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        cv_mse_scores = cross_val_score(model_cv, self.X, self.y, cv=kf, scoring=mse_scorer)
        self.cv_mse_scores = -cv_mse_scores

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_mse(self):
        print("Mean Squared Error (MSE):", self.mse)

    def _report_rmse(self):
        print("Root Mean Squared Error (RMSE):", self.rmse)

    def _report_mae(self):
        print("Mean Absolute Error (MAE):", self.mae)

    def _report_r2(self):
        print("R² (Coefficient of Determination):", self.r2)
    
    def _report_adjusted_r2(self):
        print("Adjusted R²:", self.adjusted_r2)
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (CROSS VALIDATION METRICS VISUALIZATION)
    # ------------------------------------------------------------------

    def _report_cv_mse(self):
        print("Cross-Validation Mean Squared Errors (MSE):", self.cv_mse_scores)
    
    def _report_cv_mean_mse(self):
        print("Mean CV MSE:", np.mean(self.cv_mse_scores))
    
    def _report_cv_sd_mse(self):
        print("Standard Deviation of CV MSE:", np.std(self.cv_mse_scores))
    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR FOR FIT)
    # ------------------------------------------------------------------
    def _rename_cols_by_regression_type(self):
        """
        Renames summary dataframe columns for univariate or multivariate linear regression.
        """
        if (self.regression_type.lower() == "uni"):
            self.summary_df.rename(columns={
                'Coefficient': 'Coefficient (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
        else:
            self.summary_df.rename(columns={
                'Coefficient': 'Coefficient (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)

    def _build_result_summary_df(self, labels: dict = None):
        """
        Builds result summary dataframe for linear regression.
        """
        summary_table = self.summary_table
        
        # Determine p-value column name (GLM uses P>|z|, OLS uses P>|t|)
        p_value_col = 'P>|z|' if 'P>|z|' in summary_table.columns else 'P>|t|'
        
        self.summary_df = summary_table[[
            'Coef.', '[0.025', '0.975]', p_value_col
        ]].reset_index()
        
        self.summary_df = self.summary_df.rename(columns={
            'index': 'Variable',
            'Coef.': 'Coefficient', 
            '[0.025': 'LowerCI', 
            '0.975]': 'UpperCI', 
            p_value_col: 'p-value'
        })
        
        self.summary_df = self._map_variable_label(self.summary_df, labels)
        self.summary_df = self.summary_df[['Variable', 'Coefficient', 'LowerCI', 'UpperCI', 'p-value']]

    # ------------------------------------------------------------------
    # PLOT GENERATION AND REPORTING
    # ------------------------------------------------------------------

    def _generate_residuals_vs_fitted_plot(self):
        return ResidualPlots.residuals_vs_fitted(
            residuals=self.model.resid_response,
            fitted_values=self.model.fittedvalues,
            title="Residuals vs Adjusted Values",
            xlabel="Adjusted Values",
            ylabel="Residuals"
        )
    
    def _report_residuals_vs_fitted_plot(self):
        fig = self._generate_residuals_vs_fitted_plot()
        fig.show()
    
    def _generate_qq_plot(self):
        return ResidualPlots.qq_plot(
        residuals=self.model.resid_response,
        title='Normality of Errors: Q-Q Plot'
        )
    
    def _report_qq_plot(self):
        fig = self._generate_qq_plot()
        fig.show()
    
    def _generate_forest_plot(self):
        # Use the summary_df BEFORE the (uni)/(multi) renaming
        # Or create a clean copy
        df = self.summary_df.copy()
        
        # Strip the (uni)/(multi) suffixes if they exist
        df.columns = df.columns.str.replace(r' \((uni|multi)\)', '', regex=True)
        
        fig = ForestPlot.plot(
            df=df,
            label_col='Variable',
            effect_col='Coefficient',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title="Forest Plot" + (f" ({self.regression_type})" if hasattr(self, 'regression_type') else ""),
            xaxis_title="Coefficient Estimate",
            null_value=0.0,
            log_scale=False
        )
        return fig
    
    def _report_forest_plot(self):
        fig = self._generate_forest_plot()
        fig.show()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (MODEL EVALUATION)
    # ------------------------------------------------------------------

    def _test_performance_metrics(self):
        self._evaluate_mse()
        self._evaluate_mae()
        self._evaluate_r2()

    def _test_assumptions(self):
        self._setup_assumption_tester()
        self._evaluate_independence_of_errors()
        self._evaluate_vif()
        self._evaluate_normality_of_errors_shapiro_wilk()
        self._evaluate_influential_outliers()
    
    def _test_cross_validation(self, n_splits):
        self._evaluate_cross_validation(n_splits)

    # ------------------------------------------------------------------
    # FAMILY PROPERTY FOR THIS REGRESSION 
    # ------------------------------------------------------------------
    @property
    def family(self):
        return sm.families.Gaussian()