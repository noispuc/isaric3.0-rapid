import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score

from regression import RAPID_BaseRegression


class RAPID_LogisticRegression(RAPID_BaseRegression):
    """
    Pipeline that enables logistic regression analysis for binary outcomes.
    This class implements logistic regression as part of the ISARIC analytical pipeline,
    and generates reports useful for clinical research applied to epidemiological contexts.

    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi", classification_threshold: float = 0.5):
        super().__init__(data,outcome_str,predictors_list,regression_type)
        #Validates that outcome is binary, will stop init if it is not, since it doesn't make sense to run this model
        #with non binary outcomes.
        self._validate_binary_outcome()
        #Events per variable below minimum check (doesn't cancel the init, but warns user.)
        self._evaluate_epv()
        if self.epv < 10:
            print(f"Events Per Variable (EPV) = {self.epv:.1f} is below the recommended threshold of 10.")
            print(f"Model estimates may be unstable.")
        self.classification_threshold = classification_threshold

    # ------------------------------------------------------------------
    # 2: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, 
                assumptions: bool = True,
                performance: bool = True,
                plots: list = None,
                cross_val: bool = False,
                cv_folds: int = 5,
                vif_threshold: float = 5.0):
        """
        Generate comprehensive model summary with diagnostics, performance metrics, and plots.
        
        Args:
            diagnostics (bool): If True, runs assumption checks (EPV, VIF, Cook's Distance)
            performance (bool): If True, calculates performance metrics (Accuracy, Confusion Matrix, Log Loss, Precision, Recall, F1 Score)
            plots (list, optional): List of plots ['forest_plot', 'roc_curve']
            cross_val (bool): Whether or not to run cross validation.
            cv_folds (int): number of cross validation folds (default: 5.0)
            vif_threshold (float): Threshold for flagging multicollinearity (default: 5.0)
        """
        super().summary()
        
        # Run each section
        if assumptions:
            self._summary_assumptions(vif_threshold)
        
        if performance:
            self._summary_performance_metrics()
        
        if plots:
            self._summary_generate_plots(plots)
        
        if cross_val:
            if (cv_folds > 1):
                self._summary_cross_validation(cv_folds)
            else:
                print("Cannot run cross validation with cv_folds <= 1.")
        
        print("\n" + "="*60)
        print("SUMMARY COMPLETE")
        print("="*60 + "\n")

    
    # ------------------------------------------------------------------
    # STATSMODEL FAMILY FOR THIS REGRESSION.
    # ------------------------------------------------------------------
    @property
    def family(self):
        return sm.families.Binomial()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOR CREATING THE RESULT DF AFTER FITTING MODEL)
    # ------------------------------------------------------------------
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

    def _build_result_summary_df(self, labels):
        """
        Builds result summary dataframe for logistic regression.
        """
        summary_table = self.summary_table
        summary_table['Odds Ratio'] = np.exp(summary_table['Coef.'])
        summary_table['IC Low'] = np.exp(summary_table['[0.025'])
        summary_table['IC High'] = np.exp(summary_table['0.975]'])
        self.summary_table = summary_table

        self.summary_df = summary_table[['Odds Ratio', 'IC Low', 'IC High', 'P>|z|']].reset_index()
        self.summary_df = self.summary_df.rename(columns={'index': 'Study', 
                                                          'Odds Ratio': 'OddsRatio',
                                                          'IC Low': 'LowerCI',
                                                          'IC High': 'UpperCI', 
                                                          'P>|z|': 'p-value'})
        self.summary_df = self._map_study_label(self.summary_df, labels)
        self.summary_df = self.summary_df[['Study', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (SUMMARY HELPERS)
    # ------------------------------------------------------------------

    def _summary_assumptions(self, vif_threshold):
        """Run all diagnostic checks"""
        print("\n" + "="*60)
        print("DIAGNOSTICS (Model Assumptions)")
        print("="*60 + "\n")
        
        self._evaluate_epv()
        self._evaluate_multicolinearity()
        self._evaluate_influential_outliers()

        # EPV
        self._report_epv()
        
        # VIF (Multicollinearity)
        self._report_multicollinearity(vif_threshold)
        
        # Influential Outliers
        self._report_influential_outliers()

    def _summary_performance_metrics(self):
        """Run all performance metrics"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print(f"(Classification Threshold: {self.classification_threshold})")
        print("="*60 + "\n")
        
        # Accuracy
        self._evaluate_accuracy_score()
        
        # Log Loss
        self._evaluate_log_loss()
        
        # Confusion Matrix
        self._generate_confusion_matrix()
        
        # Precision, Recall, F1
        self._evaluate_preicision()
        self._evaluate_recall_score()
        self._evaluate_f1_score()

        self._report_accuracy()
        self._report_logloss()
        print("\n")
        self._report_precision()
        self._report_recall()
        self._report_f1()
        self._report_confusion_matrix()

    def _summary_generate_plots(self, plots):
        """Generate requested plots"""
        print("\n" + "="*60)
        print("VISUALIZATIONS")
        print("="*60 + "\n")
    
        if 'forest_plot' in plots:
            self._report_forest_plot()
        
        if 'roc_curve' in plots:
            self._report_roc_curve()

    def _summary_cross_validation(self, cv_folds):
        """Run cross-validation"""
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION ({cv_folds}-Fold)")
        print("="*60 + "\n")
        
        self._evaluate_cross_validation(cv_folds)

        self._report_cv_scores()
        self._report_cv_mean()
        self._report_cv_std()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTIONS)
    # ------------------------------------------------------------------
    def _validate_binary_outcome(self):
        """Validates that outcome variable is binary and coded as 0/1"""
        # Get unique values (excluding NaN)
        unique_values = self.data[self.outcome_str].dropna().unique()
        
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
        """Calculate Events Per Variable (EPV)"""
        n_events = self.data[self.outcome_str].sum()
        n_predictors = len(self.predictors_list)
        
        if n_predictors == 0:
            raise ValueError("Cannot calculate EPV with zero predictors")
        
        self.epv = n_events/n_predictors

    def _evaluate_influential_outliers(self):
        X = pd.get_dummies(self.data[self.predictors_list], drop_first=True)
        X = sm.add_constant(X)
        X = X.astype(int)

        y = self.data[self.outcome_str].astype(float)

        model = sm.Logit(y, X).fit(maxiter=100, disp=False)

        influence = model.get_influence()

        self.cooks_d = influence.cooks_distance[0]
        self.leverage = influence.hat_matrix_diag
        self.pearson_resid = model.resid_pearson
        self.deviance_resid = model.resid_dev
        self.dfbetas = influence.dfbetas
        self.model_nobs = model.nobs

        self.influential_points = np.where(self.cooks_d > (4 / model.nobs))[0]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS)
    # ------------------------------------------------------------------
    def _evaluate_accuracy_score(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)
        self.accuracy = accuracy_score(y, y_pred_class)
    
    def _evaluate_log_loss(self):
        y = self.data[self.outcome_str]
        self.logloss = log_loss(y, self.model_result.predict())
    
    def _evaluate_preicision(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)
        self.precision = precision_score(y, y_pred_class, zero_division=0)

    def _evaluate_recall_score(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)
        self.recall = recall_score(y, y_pred_class, zero_division=0)

    def _evaluate_f1_score(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)
        self.f1 = f1_score(y, y_pred_class, zero_division=0)

    def _evaluate_cross_validation(self, cv_folds):
        X_cv = pd.get_dummies(self.data[self.predictors_list], drop_first=True).astype(float)
        y_cv = self.data[self.outcome_str].astype(int)

        clf = LogisticRegression(max_iter=1000)
        self.cross_val_scores = cross_val_score(clf, X_cv, y_cv, cv=cv_folds, scoring="accuracy")

    def _generate_roc_curve(self):
        y = self.data[self.outcome_str].astype(float)
        y_scores = self.model_result.predict()
        self.auc = roc_auc_score(y, y_scores)
        print(f"ROC AUC Score: {self.auc:.3f}")

        fpr, tpr, thresholds = roc_curve(y, y_scores)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {self.auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        return fig

    def _generate_confusion_matrix(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)

        self.cm = confusion_matrix(y, y_pred_class)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (REPORTS FOR SUMMARY)
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

    def _report_cv_scores(self):
        print("Cross-Validation Accuracies:", self.cross_val_scores)

    def _report_cv_mean(self):
        print("Mean Accuracy:", self.cross_val_scores.mean())

    def _report_cv_std(self):
        print("Standard Deviation:", self.cross_val_scores.std())

    def _report_cv_scores(self):
        print("Cross-Validation Accuracies:", self.cross_val_scores)

    def _report_cv_mean(self):
        print("Mean Accuracy:", self.cross_val_scores.mean())

    def _report_cv_std(self):
        print("Standard Deviation:", self.cross_val_scores.std())

    def _report_confusion_matrix(self):
        print("Confusion Matrix:\n", self.cm)

    def _report_roc_curve(self):
        fig = self._generate_roc_curve()
        fig.show()
    
    def _report_epv(self):
        print(f"Events Per Variable (EPV): {self.epv:.2f}")
        if self.epv < 10:
            print("Warning: EPV < 10 may lead to unstable estimates\n")
    
    def _report_influential_outliers(self):
        print(f"Number of influential points (Cook's D > 4/n): {len(self.influential_points)}")
        print("Top 5 Cook's distances:")
        print(self.cooks_d[self.influential_points][:5])
    