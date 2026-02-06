import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import KFold


from isaric.pipelines.modules.rapid_plots import ROCPlot, ForestPlot, ConfusionMatrixPlot
from isaric.pipelines.regression import RAPID_BaseRegression

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
            self._report_assumptions(vif_threshold)
        if (performance):
            self._report_performance()
        if (cross_validation):
            if not hasattr(self, 'cross_val_scores') or self.cross_val_scores is None:
                print("Cross validation not performed after fit, cannot show results.")
            else:
                self._report_cv_metrics()
        if plots is not None:
            if('forest_plot' in plots):
                self._forest_plot()
            if('roc_curve' in plots):
                self._roc_curve()
            if('confusion_matrix' in plots):
                self._confusion_matrix()
    
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

    def _evaluate_cross_validation(self, n_splits):
        """
        Perform k-fold cross-validation using the same statsmodels GLM.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_accuracy_scores = []
        
        for train_idx, test_idx in kf.split(self.X):
            # Split data
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            
            # Fit statsmodels GLM on training fold
            model_fold = sm.GLM(endog=y_train, exog=X_train, family=self.family)
            result_fold = model_fold.fit()
            
            # Predict on test fold
            y_pred_prob = result_fold.predict(X_test)
            y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
            
            # Calculate accuracy for this fold
            accuracy = accuracy_score(y_test, y_pred_class)
            cv_accuracy_scores.append(accuracy)
        
        self.cross_val_scores = np.array(cv_accuracy_scores)
    
    def _evaluate_auc_score(self):
        self.auc = roc_auc_score(self.y, self.model.fittedvalues)
    
    def _evaluate_confusion_matrix(self):
        # X and y are already preprocessed for the model
        y_pred_prob = self.model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)

        self.cm = confusion_matrix(self.y, y_pred_class)

    def _build_performance_metrics_df(self):
        """
        Build a dataframe containing all performance metrics for logistic regression.
        """
        performance_data = {
            'Metric': [
                'Accuracy',
                'Log Loss',
                'Precision',
                'Recall',
                'F1 Score',
                'AUC-ROC'
            ],
            'Value': [
                f"{self.accuracy:.6f}",
                f"{self.logloss:.6f}",
                f"{self.precision:.6f}",
                f"{self.recall:.6f}",
                f"{self.f1:.6f}",
                f"{self.auc:.6f}"
            ]
        }
        
        self.performance_metrics_df = pd.DataFrame(performance_data)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS VISUALIZATIONS)
    # ------------------------------------------------------------------
    def _report_performance(self):
        """
        Report performance metrics as a formatted dataframe.
        """
        if self.performance_metrics_df is None:
            self._build_performance_metrics_df()
        
        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        print(self.performance_metrics_df.to_string(index=False))
        print("=" * 80)
        
        # Also report confusion matrix summary
        if hasattr(self, 'cm'):
            print("\n")
            print("=" * 80)
            print("CONFUSION MATRIX")
            print("=" * 80)
            print(f"True Negatives:  {self.cm[0, 0]}")
            print(f"False Positives: {self.cm[0, 1]}")
            print(f"False Negatives: {self.cm[1, 0]}")
            print(f"True Positives:  {self.cm[1, 1]}")
            print("=" * 80)

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

    def _build_assumption_metrics_df(self, vif_threshold: float = 5.0):
        """
        Build a dataframe containing assumption test metrics for logistic regression.
        """
        # Truncate influential points to first 5
        influential_points_display = self.influential_points[:5]
        if len(self.influential_points) > 5:
            influential_points_str = f"{list(influential_points_display)} ... ({len(self.influential_points)} total)"
        else:
            influential_points_str = str(list(self.influential_points))
        
        assumption_data = {
            'Test': [
                'Events Per Variable (EPV)',
                'Influential Outliers Threshold',
                'Number of Influential Points'
            ],
            'Value': [
                f"{self.epv:.2f}",
                f"{self.influential_outliers_threshold:.6f}",
                len(self.influential_points)
            ],
            'Interpretation': [
                'Warning: EPV < 10 may lead to unstable estimates' if self.epv < 10 else 'Acceptable',
                f"Points above threshold: {influential_points_str}",
                ''
            ]
        }
        
        self.assumption_metrics_df = pd.DataFrame(assumption_data)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (CROSS VALIDATION METRICS VISUALIZATION)
    # ------------------------------------------------------------------
    def _report_cv_metrics(self):
        """
        Report cross-validation metrics as a formatted dataframe.
        """
        cv_data = {
            'Metric': [
                'Mean Accuracy',
                'Standard Deviation',
                'Individual Fold Accuracies'
            ],
            'Value': [
                f"{self.cross_val_scores.mean():.6f}",
                f"{self.cross_val_scores.std():.6f}",
                ', '.join([f"{score:.6f}" for score in self.cross_val_scores])
            ]
        }
        
        cv_df = pd.DataFrame(cv_data)
        
        print("=" * 80)
        print("CROSS-VALIDATION METRICS")
        print("=" * 80)
        print(cv_df.to_string(index=False))
        print("=" * 80)

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
    def _confusion_matrix(self):
        """Generate and display confusion matrix plot."""
        fig = ConfusionMatrixPlot.plot(
            confusion_matrix=self.cm,
            class_names=['Negative', 'Positive'],
            title='Confusion Matrix'
        )
        fig.show()

    def _forest_plot(self):
        """Generate and display forest plot for odds ratios."""
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
        fig.show()

    def _roc_curve(self):
        """Generate and display ROC curve."""
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
        self._build_performance_metrics_df()

    def _test_assumptions(self):
        self._setup_assumption_tester()
        self._evaluate_vif()
        self._evaluate_influential_outliers()
        self._evaluate_epv()
        self._build_assumption_metrics_df()
    
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