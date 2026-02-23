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

    def __init__(self, data: pd.DataFrame, yvar: str = None, predictors: list = None,
                formula: str = None, family:str = "binomial", link:str = "logit", 
                regression_type: str = "Multi", classification_threshold: float = 0.5):
        super().__init__(data=data, yvar=yvar, predictors=predictors, formula=formula, 
                        family=family, link=link, regression_type=regression_type)
        self.classification_threshold = classification_threshold

    def summary(self, assumptions=None, performance=None, cross_val=None,
                plots: list = None, vif_threshold: float = 5.0):
        """
        Reports the results of the logistic regression, generating tables and plots.

        Args:
            assumptions: 'all' shows everything. A list of strings selects specific metrics.
                        Available: 'Events Per Variable (EPV)', 'Influential Outliers Threshold',
                        'Number of Influential Points', 'VIF', 'Influential Outliers'. None skips.
            performance: 'all' shows everything. A list of strings selects specific metrics.
                        Available: 'Accuracy', 'Log Loss', 'Precision', 'Recall', 'F1 Score',
                        'AUC-ROC', 'AIC', 'BIC', 'LLF', 'McFadden R2', 'Adjusted McFadden R2',
                        'Efron R2', 'Cox Snell R2', 'Nagelkerke R2', 'Tjur R2',
                        'Confusion Matrix'. None skips.
            cross_val:   'all' shows everything. A list of strings selects specific metrics.
                        Available: 'Mean Accuracy', 'Standard Deviation',
                        'Individual Fold Accuracies'. None skips.
            plots (list): List of plots ['forest_plot', 'roc_curve', 'confusion_matrix']
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
            if not hasattr(self, 'cross_val_scores') or self.cross_val_scores is None:
                print("Cross validation not performed after fit, cannot show results.")
            else:
                self._report_cv_metrics(metrics=cross_validation)
        if plots is not None:
            if 'forest_plot' in plots:
                self._forest_plot()
            if 'roc_curve' in plots:
                self._roc_curve()
            if 'confusion_matrix' in plots:
                self._confusion_matrix()
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS EVALUATION)
    # ------------------------------------------------------------------
    def _evaluate_accuracy_score(self):
        y_pred_prob = self.fitted_model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.accuracy = accuracy_score(self.y, y_pred_class)
    
    def _evaluate_log_loss(self):
        y_pred_prob = self.fitted_model.fittedvalues
        self.logloss = log_loss(self.y, y_pred_prob)
    
    def _evaluate_r2(self):
        n = int(self.fitted_model.nobs)
        ll_model = self.fitted_model.llf
        ll_null = self.fitted_model.llnull

        self.cox_snell_r2 = 1 - np.exp((2 / n) * (ll_null - ll_model))
        self.nagelkerke_r2 = self.cox_snell_r2 / (1 - np.exp((2 / n) * ll_null))

        y_array = np.asarray(self.y).ravel()
        fitted = np.asarray(self.fitted_model.fittedvalues).ravel()
        self.tjur_r2 = fitted[y_array == 1].mean() - fitted[y_array == 0].mean()

    def _evaluate_precision(self):
        y_pred_prob = self.fitted_model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.precision = precision_score(self.y, y_pred_class, zero_division=0)

    def _evaluate_recall_score(self):
        y_pred_prob = self.fitted_model.fittedvalues
        y_pred_class = (y_pred_prob >= self.classification_threshold).astype(int)
        self.recall = recall_score(self.y, y_pred_class, zero_division=0)

    def _evaluate_f1_score(self):
        y_pred_prob = self.fitted_model.fittedvalues
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
        self.auc = roc_auc_score(self.y, self.fitted_model.fittedvalues)
    
    def _evaluate_confusion_matrix(self):
        # X and y are already preprocessed for the model
        y_pred_prob = self.fitted_model.fittedvalues
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
                'AUC-ROC',
                'AIC',
                'BIC',
                'LLF',
                'McFadden R2',
                'Adjusted McFadden R2',
                'Efron R2',
                'Cox Snell R2',
                'Nagelkerke R2',
                'Tjur R2'
            ],
            'Value': [
                f"{self.accuracy:.6f}",
                f"{self.logloss:.6f}",
                f"{self.precision:.6f}",
                f"{self.recall:.6f}",
                f"{self.f1:.6f}",
                f"{self.auc:.6f}",
                f"{self.aic:.6f}",
                f"{self.bic:.6f}",
                f"{self.llf:.6f}",
                f"{self.mcfadden_r2}",
                f"{self.mcfadden_adj_r2}",
                f"{self.efron_r2}",
                f"{self.cox_snell_r2}",
                f"{self.nagelkerke_r2}",
                f"{self.tjur_r2}"

            ]
        }
        
        self.performance_metrics_df = pd.DataFrame(performance_data)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS VISUALIZATIONS)
    # ------------------------------------------------------------------
    def _report_performance(self, metrics=None):
        if self.performance_metrics_df is None:
            self._build_performance_metrics_df()

        df = self.performance_metrics_df
        if metrics != 'all':
            missing = set(metrics) - set(df['Metric']) - {'Confusion Matrix'}
            if missing:
                print(f"Warning: the following performance metrics were not found: {missing}")
            df = df[df['Metric'].isin(metrics)]

        print("=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        if (metrics == 'all' or 'Confusion Matrix' in metrics) and hasattr(self, 'cm'):
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
    def _validate_binary_outcome(self, data, yvar, formula):
        """Validates that outcome variable is binary and coded as 0/1"""
        if yvar:
            outcome = data[yvar].dropna()
        elif formula:
            # Extract outcome variable name from left-hand side of formula
            outcome_name = formula.split('~')[0].strip()
            outcome = data[outcome_name].dropna()
        else:
            return

        unique_values = outcome.unique()

        if len(unique_values) != 2:
            raise ValueError(
                f"Logistic regression requires a binary outcome variable. "
                f"Found {len(unique_values)} unique values: {unique_values}"
            )

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
    def _build_cv_df(self):
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
        y_scores = self.fitted_model.fittedvalues
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
        super()._test_performance_metrics()
        self._evaluate_r2()
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
        self._build_cv_df()


    # ------------------------------------------------------------------
    # NECESSARY DATA VALIDATIONS BEFORE PREPROCESSING
    # ------------------------------------------------------------------
    def _run_data_validations(self,data, yvar, predictors, formula, family, link, regression_type):
        super()._run_data_validations(data, yvar, predictors, formula, family, link, regression_type)
        self._validate_binary_outcome(data, yvar, formula)

    # ------------------------------------------------------------------
    # FAMILY AND LINK MAPS
    # ------------------------------------------------------------------
    @property
    def _family_map(self):
        return {
            "binomial": sm.families.Binomial
            }
    
    @property
    def _link_map(self):
        return {
            "logit":  sm.families.links.Logit,
            "probit": sm.families.links.Probit,
            "cloglog": sm.families.links.CLogLog,  # complementary log-log
            "log":    sm.families.links.Log,  
        }