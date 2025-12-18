import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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
        self.epv = self._calculate_epv()
        if self.epv < 10:
            print(f"Events Per Variable (EPV) = {self.epv:.1f} is below the recommended threshold of 10.")
            print(f"Model estimates may be unstable.")
        self.classification_threshold = classification_threshold

    # ------------------------------------------------------------------
    # 2: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, 
                diagnostics: bool = True,
                performance: bool = True,
                plots: list = None,
                cv_folds: int = None,
                classification_threshold: float = None,
                vif_threshold: float = 5.0,
                print_results: bool = True):
        """
        Generate comprehensive model summary with diagnostics, performance metrics, and plots.
        
        Args:
            diagnostics (bool): If True, runs assumption checks (EPV, VIF, Cook's Distance)
            performance (bool): If True, calculates performance metrics (Accuracy, Log Loss, etc.)
            plots (list, optional): List of plots ['forest_plot', 'roc_curve']
            cv_folds (int, optional): Number of folds for cross-validation
            classification_threshold (float, optional): Override default classification threshold
            vif_threshold (float): Threshold for flagging multicollinearity (default: 5.0)
            print_results (bool): If True, prints results to console (default: True)
        """
        super().summary()
        
        # Handle threshold override
        current_threshold = self._setup_threshold(classification_threshold)
        
        # Run each section
        if diagnostics:
            self._run_diagnostics(vif_threshold, print_results)
        
        if performance:
            self._run_performance_metrics(print_results)
        
        if plots:
            self._generate_plots(plots, print_results)
        
        if cv_folds:
            self._run_cross_validation(cv_folds, print_results)
        
        # Restore original threshold
        self.classification_threshold = current_threshold
        
        if print_results:
            print("\n" + "="*60)
            print("SUMMARY COMPLETE")
            print("="*60 + "\n")
            super().summary()

    
    # ------------------------------------------------------------------
    # STATSMODEL FAMILY FOR THIS REGRESSION.
    # ------------------------------------------------------------------
    def family(self):
        return sm.families.Binomial

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
    # PRIVATE METHODS (PLOTS FOR SUMMARY)
    # ------------------------------------------------------------------
    def _generate_roc_curve(self):
        y = self.data[self.outcome_str].astype(float)
        y_scores = self.model_result.predict()
        self.auc = roc_auc_score(y, y_scores)
        print(f"ROC AUC Score: {self.auc:.3f}")


        fpr, tpr, thresholds = roc_curve(y, y_scores)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()
    
    def _show_confusion_matrix(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)

        self.cm = confusion_matrix(y, y_pred_class)
        print("Confusion Matrix:\n", self.cm)

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

    def _calculate_epv(self):
        """Calculate Events Per Variable (EPV)"""
        n_events = self.data[self.outcome_str].sum()
        n_predictors = len(self.predictors_list)
        
        if n_predictors == 0:
            raise ValueError("Cannot calculate EPV with zero predictors")
        
        return n_events / n_predictors
    
    def _multicolinearity_validation(self):
        """Check whether independent variables are perfectly correlated with each other."""
        X = pd.get_dummies(self.data[self.predictors_list], drop_first=True)
        X = sm.add_constant(X)

        X = X.astype(int)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
        self.vif_data = vif_data[vif_data["Variable"] != "const"]

    def _influential_outliers_validation(self):
        X = pd.get_dummies(self.data[self.predictors_list], drop_first=True)
        X = sm.add_constant(X)

        y = self.data[self.outcome_str].astype(float)

        model = sm.Logit(y, X).fit(maxiter=100, disp=False)

        influence = model.get_influence()

        self.cooks_d = influence.cooks_distance[0]
        self.leverage = influence.hat_matrix_diag
        self.pearson_resid = model.resid_pearson
        self.deviance_resid = model.resid_dev
        self.dfbetas = influence.dfbetas
        self.model_nobs = model.nobs

        self.influential_points = np.where(
            self.cooks_d > 4 / model.nobs
        )[0]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PERFORMANCE METRICS)
    # ------------------------------------------------------------------
    def _evaluate_accuracy_score(self):
        y = self.data[self.outcome_str]
        y_pred_class = (self.model_result.predict() >= self.classification_threshold).astype(int)
        self.accuracy = accuracy_score(y, y_pred_class)
        print("Accuracy:", self.accuracy)
    
    def _evaluate_log_loss(self):
        y = self.data[self.outcome_str]
        self.logloss = log_loss(y, self.model_result.predict())
        print("Log Loss:", self.logloss)
    
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

    def _evauluate_cross_validation(self):
        X_cv = pd.get_dummies(self.data[[self.predictors_list]], drop_first=True).astype(float)
        y_cv = self.data[self.outcome_str].astype(int)

        clf = LogisticRegression(max_iter=1000)
        self.cross_val_scores = cross_val_score(clf, X_cv, y_cv, cv=5, scoring="accuracy")

    # ------------------------------------------------------------------
    # PRIVATE METHODS (SUMMARY HELPERS)
    # ------------------------------------------------------------------

    def _setup_threshold(self, classification_threshold):
        """Setup and return the threshold to use for this summary call"""
        if classification_threshold is not None:
            original_threshold = self.classification_threshold
            self.classification_threshold = classification_threshold
            return original_threshold
        return self.classification_threshold

    def _run_diagnostics(self, vif_threshold, print_results):
        """Run all diagnostic checks"""
        if print_results:
            print("\n" + "="*60)
            print("DIAGNOSTICS (Model Assumptions)")
            print("="*60 + "\n")
        
        # EPV
        self._report_epv(print_results)
        
        # VIF (Multicollinearity)
        self._report_multicollinearity(vif_threshold, print_results)
        
        # Influential Outliers
        self._report_influential_outliers(print_results)

    def _report_epv(self, print_results):
        """Report Events Per Variable"""
        if print_results:
            print(f"Events Per Variable (EPV): {self.epv:.2f}")
            if self.epv < 10:
                print("Warning: EPV < 10 may lead to unstable estimates\n")

    def _report_multicollinearity(self, vif_threshold, print_results):
        """Report VIF and flag problematic variables"""
        self._multicolinearity_validation()
        if print_results:
            print("\nVariance Inflation Factor (VIF):")
            print(self.vif_data)
        
        problematic_vif = self.vif_data[self.vif_data['VIF'] > vif_threshold]
        
        if print_results:
            if not problematic_vif.empty:
                print(f"\nVariables with VIF > {vif_threshold}:")
                print(problematic_vif)
            else:
                print(f"\nNo variables with VIF > {vif_threshold}")

    def _report_influential_outliers(self, print_results):
        """Report Cook's Distance and influential points"""
        if print_results:
            print("\nInfluential Outliers (Cook's Distance):")
        
        self._influential_outliers_validation()
        
        threshold = 4 / self.model_nobs
        
        if print_results:
            print(f"Number of influential points: {len(self.influential_points)}")
            print(f"Threshold (4/n): {threshold:.4f}")
            if len(self.influential_points) > 0:
                preview = self.influential_points[:10]
                suffix = '...' if len(self.influential_points) > 10 else ''
                print(f"Influential point indices: {preview}{suffix}")

    def _run_performance_metrics(self, print_results):
        """Run all performance metrics"""
        if print_results:
            print("\n" + "="*60)
            print("PERFORMANCE METRICS")
            print(f"(Classification Threshold: {self.classification_threshold})")
            print("="*60 + "\n")
        
        # Accuracy
        self._evaluate_accuracy_score()
        
        # Log Loss
        self._evaluate_log_loss()
        
        # Confusion Matrix
        if print_results:
            print("\nConfusion Matrix:")
        self._show_confusion_matrix()
        
        # Precision, Recall, F1
        self._evaluate_preicision()
        self._evaluate_recall_score()
        self._evaluate_f1_score()

        if(print_results):
            print("Precision:", round(self.precision, 4))
            print("Recall:", round(self.recall, 4))
            print("F1 Score:", round(self.f1, 4))

    def _generate_plots(self, plots, print_results):
        """Generate requested plots"""
        if print_results:
            print("\n" + "="*60)
            print("VISUALIZATIONS")
            print("="*60 + "\n")
        
        if 'forest_plot' in plots:
            self._generate_forest_plot(print_results)
        
        if 'roc_curve' in plots:
            self._generate_roc_curve_plot(print_results)

    def _generate_forest_plot(self, print_results):
        """Generate forest plot using parent class method"""
        if print_results:
            print("Generating Forest Plot...")
        
        self._display_forest_plot()

    def _generate_roc_curve_plot(self, print_results):
        """Generate ROC curve"""
        if print_results:
            print("\nGenerating ROC Curve...")
        
        self._generate_roc_curve()

    def _run_cross_validation(self, cv_folds, print_results):
        """Run cross-validation"""
        if print_results:
            print("\n" + "="*60)
            print(f"CROSS-VALIDATION ({cv_folds}-Fold)")
            print("="*60 + "\n")
        
        self._evauluate_cross_validation()

        if (print_results):
            print("Cross-Validation Accuracies:", self.cross_val_scores)
            print("Mean Accuracy:", self.cross_val_scores.mean())
            print("Standard Deviation:", self.cross_val_scores.std())