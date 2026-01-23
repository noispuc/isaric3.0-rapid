import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, cross_val_score

from statsmodels.stats.stattools import durbin_watson

from isaric.pipelines.modules.rapid_plots import ROCPlot, ForestPlot, ConfusionMatrixPlot
from isaric.pipelines.regression import RAPID_BaseRegression
from isaric.pipelines.modules.rapid_assumption import ModelAssumptionTester

class RAPID_LogisticRegression(RAPID_BaseRegression):

    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi", classification_threshold: float = 0.5):
        super().__init__(data,outcome_str,predictors_list,regression_type)
        self.classification_threshold = classification_threshold

    def summary(self, assumptions: bool = True, performance: bool = True,
                plots: list = None,
                cross_val: bool = False,
                vif_threshold: float = 5.0):
        self._visualization(assumptions,performance,cross_val,plots,vif_threshold)
    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------

    def _validation():
        pass

    def _visualization(self, assumptions: bool = True, performance: bool = False, cross_validation: bool = False, 
                       plots: list = None, vif_threshold: float = 5.0):
        if (assumptions):
            self._report_vif(vif_threshold)
            self._report_epv()
            self._report_influential_outliers()
        if (performance):
            self._report_accuracy()
            self._report_logloss()
            self._report_precision()
            self._report_recall()
            self._report_f1()
        if (cross_validation):
            if not hasattr(self, 'cross_val_scores') or self.cross_val_scores is None:
                print("Cross validation not performed after fit, cannot show results.")
            else:
                self._report_cv_scores()
                self._report_cv_mean()
                self._report_cv_std()
        if plots is not None:
            if('forest_plot' in plots):
                self._report_forest_plot()
            if('roc_curve' in plots):
                self._report_roc_curve()
            if('confusion_matrix' in plots):
                self._report_confusion_matrix()
    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS EVALUATION)
    # ------------------------------------------------------------------
    def _evaluate_accuracy_score(self):
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.accuracy = accuracy_score(self.y, y_pred_class)
    
    def _evaluate_log_loss(self):
        y_pred_prob = self.model.fittedvalues
        self.logloss = log_loss(self.y, y_pred_prob)

    def _evaluate_precision(self):
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.precision = precision_score(self.y, y_pred_class, zero_division=0)

    def _evaluate_recall_score(self):
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.recall = recall_score(self.y, y_pred_class, zero_division=0)

    def _evaluate_f1_score(self):
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.f1 = f1_score(self.y, y_pred_class, zero_division=0)

    def _evaluate_cross_validation(self, n_folds):
        clf = LogisticRegression(max_iter=1000)
        self.cross_val_scores = cross_val_score(
            clf, 
            self.X, 
            self.y, 
            cv=n_folds, 
            scoring="accuracy"
        )
    
    def _evaluate_auc_score(self):
        self.auc = roc_auc_score(self.y, self.model.fittedvalues)
    
    def _evaluate_confusion_matrix(self):
        # X and y are already preprocessed for the model
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)

        self.cm = confusion_matrix(self.y, y_pred_class)


    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS VISUALIZATIONS)
    # ------------------------------------------------------------------
    def _report_accuracy(self):
        print("Accuracy:", self.accuracy)

    def _report_logloss(self):
        print("Log Loss:", self.logloss)

    def _report_precision(self):
        print("Precision:", round(self.precision, 4))

    def _report_recall(self):
        print("Recall:", round(self.recall, 4))

    def _report_f1(self):
        print("F1 Score:", round(self.f1, 4))
    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTIONS EVALUATION)
    # ------------------------------------------------------------------
    def _validate_binary_outcome(self, data, outcome_str):
        """Validates that outcome variable is binary and coded as 0/1"""
        # Get unique values (excluding NaN)
        unique_values = data[outcome_str].dropna().unique()
        
        # Check if binary (exactly 2 unique values)
        if len(unique_values) != 2:
            raise ValueError(
                f"Logistic regression requires a binary outcome variable. "
                f"Found {len(unique_values)} unique values: {unique_values}"
            )
        
        # Check if coded as 0 and 1
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(
                f"Outcome variable must be coded as 0 and 1. "
                f"Found values: {sorted(unique_values)}"
            )
        
    def _evaluate_epv(self):
        epv_result = self.assumption_tester.test_epv()
        self.epv = epv_result['epv']

    def _evaluate_influential_outliers(self):
        influence = self.model.get_influence()
        
        self.cooks_d = influence.cooks_distance[0]
        self.leverage = influence.hat_matrix_diag
        self.pearson_resid = self.model.resid_pearson
        self.deviance_resid = self.model.resid_deviance
        self.dfbetas = influence.dfbetas
        self.model_nobs = self.model.nobs
        self.influential_outliers_threshold = 4/self.model.nobs
        
        self.influential_points = np.where(self.cooks_d > (4 / self.model.nobs))[0]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTIONS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_epv(self):
        print(f"Events Per Variable (EPV): {self.epv:.2f}")
        if self.epv < 10:
            print("Warning: EPV < 10 may lead to unstable estimates\n")

    # ------------------------------------------------------------------
    # PRIVATE METHODS (CROSS VALIDATION METRICS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_cv_scores(self):
        print("Cross-Validation Accuracies:", self.cross_val_scores)

    def _report_cv_mean(self):
        print("Mean Accuracy:", self.cross_val_scores.mean())

    def _report_cv_std(self):
        print("Standard Deviation:", self.cross_val_scores.std())

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOR CREATING THE RESULT DF AFTER FITTING MODEL)
    # ------------------------------------------------------------------
    def _build_result_summary_df(self, labels):
        """
        Builds result summary dataframe for logistic regression.
        """
        summary_table = self.summary_table
        
        # Determine p-value column name (GLM uses P>|z|)
        p_value_col = 'P>|z|' if 'P>|z|' in summary_table.columns else 'P>|t|'
        
        # Calculate odds ratios and confidence intervals
        summary_table['Odds Ratio'] = np.exp(summary_table['Coef.'])
        summary_table['IC Low'] = np.exp(summary_table['[0.025'])
        summary_table['IC High'] = np.exp(summary_table['0.975]'])
        
        # Select and rename columns
        self.summary_df = summary_table[['Odds Ratio', 'IC Low', 'IC High', p_value_col]].reset_index()
        self.summary_df = self.summary_df.rename(columns={
            'index': 'Variable',  # Changed from 'Study' to 'Variable' for consistency
            'Odds Ratio': 'OddsRatio',
            'IC Low': 'LowerCI',
            'IC High': 'UpperCI', 
            p_value_col: 'p-value'
        })
        
        # Apply variable labels
        self.summary_df = self._map_variable_label(self.summary_df, labels)
        self.summary_df = self.summary_df[['Variable', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]

    def _rename_cols_by_regression_type(self):
        """
        Renames summary dataframe columns for univariate or multivariate logistic regression.
        """
        if (self.regression_type.lower() == "uni"):
            self.summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (uni)',
                'LowerCI': 'LowerCI (uni)',
                'UpperCI': 'UpperCI (uni)',
                'p-value': 'p-value (uni)'
            }, inplace=True)
        else:
            self.summary_df.rename(columns={
                'OddsRatio': 'OddsRatio (multi)',
                'LowerCI': 'LowerCI (multi)',
                'UpperCI': 'UpperCI (multi)',
                'p-value': 'p-value (multi)'
            }, inplace=True)
    # ------------------------------------------------------------------
    # PLOT GENERATION AND REPORTING
    # ------------------------------------------------------------------
    def _generate_confusion_matrix(self):
        return ConfusionMatrixPlot.plot(
        confusion_matrix=self.cm,
        class_names=['Negative', 'Positive'],
        title='Confusion Matrix'
        )

    def _report_confusion_matrix(self):
        fig = self._generate_confusion_matrix()
        fig.show()

    def _generate_forest_plot(self):
        """Generate forest plot for odds ratios."""
        df = self.summary_df.copy()
        
        # Strip the (uni)/(multi) suffixes if they exist
        df.columns = df.columns.str.replace(r' \((uni|multi)\)', '', regex=True)
        
        fig = ForestPlot.plot(
            df=df,
            label_col='Variable',
            effect_col='OddsRatio',
            lower_col='LowerCI',
            upper_col='UpperCI',
            title=f"Forest Plot - Odds Ratios ({self.regression_type})" if hasattr(self, 'regression_type') else "Forest Plot - Odds Ratios",
            xaxis_title="Odds Ratio",
            null_value=1.0,  # For OR, null value is 1 (not 0!)
            log_scale=True   # Use log scale for odds ratios
        )
        return fig

    def _report_forest_plot(self):
        fig = self._generate_forest_plot()
        fig.show()

    def _generate_roc_curve(self):
        y_scores = self.model.fittedvalues
        fpr, tpr, _ = roc_curve(self.y, y_scores)
        self.auc = roc_auc_score(self.y, y_scores)

        fig = ROCPlot.plot(
            fpr=fpr,
            tpr=tpr,
            auc=self.auc,
            title='ROC Curve - Logistic Regression',
            label=f'Logistic Regression (AUC = {self.auc:.3f})'
        )

        return fig

    def _report_roc_curve(self):
        fig = self._generate_roc_curve()
        fig.show()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (MODEL EVALUATION)
    # ------------------------------------------------------------------

    def _test_performance_metrics(self):
        self._evaluate_accuracy_score()
        self._evaluate_log_loss()
        self._evaluate_precision()
        self._evaluate_recall_score()
        self._evaluate_f1_score()
        self._evaluate_auc_score()
        self._evaluate_confusion_matrix()

    def _test_assumptions(self):
        self._setup_assumption_tester()
        self._evaluate_vif()
        self._evaluate_influential_outliers()
        self._evaluate_epv()
    
    def _test_cross_validation(self, n_splits):
        self._evaluate_cross_validation(n_splits)


    # ------------------------------------------------------------------
    # NECESSARY DATA VALIDATIONS BEFORE PREPROCESSING
    # ------------------------------------------------------------------
    def _run_data_validations(self, data, outcome_str, predictors_list, regression_type):
        super()._run_data_validations(data, outcome_str, predictors_list, regression_type)
        self._validate_binary_outcome(data, outcome_str)

    # ------------------------------------------------------------------
    # STATSMODEL FAMILY FOR THIS REGRESSION.
    # ------------------------------------------------------------------
    @property
    def family(self):
        return sm.families.Binomial()