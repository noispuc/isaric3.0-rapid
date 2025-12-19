import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_val_score
from statsmodels.stats.stattools import durbin_watson

from regression import RAPID_BaseRegression

class RAPID_LinearRegression(RAPID_BaseRegression):

    """
    Pipeline that enables linear regression analysis for continuous outcomes.
    This class implements linear regression as part of the ISARIC analytical pipeline,
    and generates reports useful for clinical research applied to epidemiological contexts.

    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        super().__init__(data,outcome_str,predictors_list,regression_type)
        
    # ------------------------------------------------------------------
    # SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, assumptions: bool = False, performance: bool = False, plots: list = None, cross_val: bool = False, k_folds: int = 5, vif_threshold: float = 5.0):
        """
        Reports the results of the linear regression, generating tables and plots.

        Args:
            diagnostics (bool): If True, runs assumption checks (Independence of Errors, Normality of Errors, Multicolinearity, Influential Outliers, Mean of Errors)
            performance (bool): If True, calculates performance metrics (MSE, RMSE, MAE, R^2, Adjusted R^2)
            plots (list, optional): List of plots ['forest_plot', 'residuals_vs_fitted', 'homoscedasticity', 'qq_plot']
            cross_val (bool): Whether or not to run cross validation.
            k_folds (int): number of cross validation folds (default: 5.0)
            vif_threshold (float): Threshold for flagging multicollinearity (default: 5.0)
        """
        super().summary()
        self._setup_validation_model()
        if (assumptions):
            self._summary_assumptions()
        
        if (performance):
            self._summary_performance(cross_val, k_folds)
        
        if (plots):
            self._summary_plots(plots)
    
    # ------------------------------------------------------------------
    # STATSMODEL FAMILY FOR THIS REGRESSION.
    # ------------------------------------------------------------------
    @property
    def family(self):
        return sm.families.Gaussian()
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOR CREATING SUMMARY DF AFTER FITTING MODEL)
    # ------------------------------------------------------------------
    def _rename_cols_by_regression_type(self):
        """
        Renames summary dataframe columns for univariate or multivariate logistic regression.
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
    
    def _build_result_summary_df(self, labels):
        """
        Builds result summary dataframe for linear regression.
        """
        summary_table = self.summary_table

        self.summary_df = summary_table[['Coef.', '[0.025', '0.975]', 'P>|z|']].reset_index()
        self.summary_df = self.summary_df.rename(columns={'index': 'Study',
                                                     'Coef.': 'Coefficient', 
                                                     '[0.025': 'LowerCI', 
                                                     '0.975]': 'UpperCI', 
                                                     'P>|z|': 'p-value'})
        
        self.summary_df = self._map_study_label(self.summary_df, labels)
        self.summary_df = self.summary_df[['Study', 'Coefficient', 'LowerCI', 'UpperCI', 'p-value']]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (SUMMARY HELPERS)
    # ------------------------------------------------------------------
    def _summary_assumptions(self):
        self._evaluate_independence_of_errors()
        self._evaluate_homoscedasticity()
        self._evaluate_normality_of_errors_shapiro_wilk()
        self._evaluate_influential_outliers()

        self._report_independence_of_errors()
        self._report_homoscedasticity()
        self._report_normality_of_errors_shapiro()
        self._report_influential_outliers()


    def _summary_performance(self, cross_val, n_splits):
        self._evaluate_mse()
        self._evaluate_mae()
        self._evaluate_r2()
        if(cross_val):
            self._evaluate_cross_validation(n_splits)

        self._report_mse()
        self._report_mae()

        self._report_r2()
        self._report_adjusted_r2()

        if (cross_val):
            self._report_cv_mse()
            self._report_cv_mean_mse()
            self._report_cv_sd_mse

    def _summary_plots(self, plots):
        if "forest_plot" in plots:
            self._report_forest_plot()
        if "residuals_vs_fitted" in plots:
            self._report_linearity()
        if "qq_plot" in plots:
            self._report_normality_of_errors_figure()
        if "homoscedasticity" in plots:
            self._report_homoscedasticity()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTIONS)
    # ------------------------------------------------------------------

    def _setup_validation_model(self):
        X = self.data[self.predictors_list]
        X = pd.get_dummies(X, drop_first=True)
        X = sm.add_constant(X)

        y = self.data[self.outcome_str]

        y = y.astype(float)
        X = X.astype(float)

        data = pd.concat([y, X], axis=1)
        data_clean = data.dropna()

        y_clean = data_clean[y.name]
        X_clean = data_clean.drop(y.name, axis=1)

        self.y_clean_validation = y_clean
        self.X_clean_validation = X_clean

        self.validation_model = sm.OLS(y_clean, X_clean).fit()
    
    def _evaluate_linearity(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.validation_model.fittedvalues,
            y=self.validation_model.resid,
            mode='markers',
            marker=dict(color='blue', size=8)
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='red')

        fig.update_layout(
            title='Residuals vs Adjusted Values',
            xaxis_title='Adjusted Values',
            yaxis_title='Residuals',
            yaxis_range=[min(self.validation_model.resid)*1.1, max(self.validation_model.resid)*1.1]
        )
        return fig

    def _evaluate_independence_of_errors(self):
        self.dw = durbin_watson(self.validation_model.resid)

    def _evaluate_homoscedasticity(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.validation_model.fittedvalues,
            y=self.validation_model.resid,
            mode='markers',
            marker=dict(color='blue', size=8)
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='red')

        fig.update_layout(
            title='Residuals vs Adjusted Values',
            xaxis_title='Adjusted Values',
            yaxis_title='Residuals',
            yaxis_range=[min(self.validation_model.resid)*1.1, max(self.validation_model.resid)*1.1]
        )
        return fig
    
    def _evaluate_normality_of_errors_qq_plot(self):
        fig = go.Figure()
        qq = stats.probplot(self.validation_model.resid, dist="norm")
        theoretical_quantiles = qq[0][0]
        ordered_residuals = qq[0][1]

        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=ordered_residuals,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Sample Quantiles'
        ))


        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Ideal Normal'
        ))

        fig.update_layout(
            title='Normality of Errors: Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
        )

        return fig

    def _evaluate_normality_of_errors_shapiro_wilk(self):
        shapiro_test = stats.shapiro(self.validation_model.resid)
        self.shapiro_wilk_test_statistic = shapiro_test.statistic
        self.shapiro_wilk_p_value = shapiro_test.pvalue

        return self.shapiro_wilk_test_statistic, self.shapiro_wilk_p_value 

    def _evaluate_influential_outliers(self):
        influence = self.validation_model.get_influence()

        self.cooks_d = influence.cooks_distance[0]

        obs = range(len(self.cooks_d))

        self.influential_outliers_threshold = 4 / len(self.cooks_d)

        self.influential_points = [i for i, val in enumerate(self.cooks_d) if val > self.influential_outliers_threshold]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS)
    # ------------------------------------------------------------------
    def _evaluate_mse(self):
        self.mse = mean_squared_error(self.y_clean_validation, self.validation_model.fittedvalues)
        #To ensure that one is consistent with the other through evaluations, this also evaluates the RMSE.
        self.rmse = np.sqrt(self.mse)
    
    def _evaluate_mae(self):
        self.mae = mean_absolute_error(self.y_clean_validation, self.validation_model.fittedvalues)
    
    def _evaluate_r2(self):
        n = self.X_clean_validation.shape[0]
        p = self.X_clean_validation.shape[1]

        self.r2 = r2_score(self.y_clean_validation, self.validation_model.fittedvalues)
        #To ensure that one is consistent with the other through evaluations, this also evaluates the adjusted R2.
        self.adjusted_r2 = 1 - (1 - self.r2) * ((n - 1) / (n - p - 1))
    
    def _evaluate_cross_validation(self, n_splits):
        model_cv = LinearRegression()

        # Define the cross-validation strategy
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Define the scoring metric (negative MSE because scikit-learn uses "higher is better" by default)
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Perform cross-validation
        cv_mse_scores = cross_val_score(model_cv, self.X_clean_validation, self.y_clean_validation, cv=kf, scoring=mse_scorer)

        # Convert scores to positive MSE
        self.cv_mse_scores = -cv_mse_scores
    # ------------------------------------------------------------------
    # PRIVATE METHODS (REPORTS FOR SUMMARY)
    # ------------------------------------------------------------------
    def _report_linearity(self):
        fig = self._evaluate_linearity()
        fig.show()
    
    def _report_independence_of_errors(self):
        print(f'Durbin-Watson Statistic: {self.dw:.3f}')
        if self.dw < 1.5:
            print("→ Indicates positive autocorrelation of residuals.")
        elif self.dw > 2.5:
            print("→ Indicates negative autocorrelation of residuals.")
        else:
            print("→ Residuals are likely independent.")

    def _report_homoscedasticity(self):
        fig = self._evaluate_homoscedasticity()
        fig.show()

    def _report_normality_of_errors_figure(self):
        fig = self._evaluate_normality_of_errors_qq_plot()
        fig.show()
    
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
    
    def _report_cv_mse(self):
        print("Cross-Validation Mean Squared Errors (MSE):", self.cv_mse_scores)
    
    def _report_cv_mean_mse(self):
        print("Mean CV MSE:", np.mean(self.cv_mse_scores))
    
    def _report_cv_sd_mse(self):
        print("Standard Deviation of CV MSE:", np.std(self.cv_mse_scores))