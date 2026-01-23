from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from isaric.pipelines.modules.rapid_preprocess import RapidPreprocessor
from isaric.pipelines.modules.rapid_assumption import ModelAssumptionTester
from .pipeline import RAPID_Pipeline



class RAPID_BaseRegression(RAPID_Pipeline):
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        self._run_data_validations(data, outcome_str, predictors_list, regression_type)
        self.data = data.copy()
        self.outcome_str = outcome_str
        self.predictors_list = predictors_list
        self.regression_type = regression_type
        self.preprocess_data()

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------
    def preprocess_data(self):
        self._data_cleaning()
        self._preprocessing()

    def fit(self, labels: dict = None, cross_val: bool = True, n_splits: int = 5):
        """
        Fits the model. 
        Calculates assumption tests, performance metrics and optionally performs cross validation.

        Args:
            labels(dict): Maps variable names to human legible names for display.
            cross_val(bool): Whether or not to perform cross validation.
            n_splits(int): Number of splits for cross validation (default: 5).

        """
        if (self.X is None or self.y is None):
            print("Please run preprocess_data before fitting model.")
        self._modeling()
        self._model_evaluation(labels=labels, cv=cross_val, splits=n_splits)

    def summary(self):
        """
        Summary to be output by this regression
        """
        if self.model is None:
            print("self.model does not exist. Fit the model before calling summary.")
            return

    # ------------------------------------------------------------------
    # ASSUMPTION TESTING (SHARED)
    # ------------------------------------------------------------------        

    def _setup_assumption_tester(self):
        self.assumption_tester = ModelAssumptionTester(model=self.model, X=self.X, y=self.y, y_pred=self.model.fittedvalues)

    def _evaluate_vif(self):
        self.vif_results = self.assumption_tester.test_vif()
        
    def _evaluate_influential_outliers(self):
        #In the rapid assumption module, there is a model agnostic computation for cook's distance
        #However, this is not used in this computation because the statsmodels implementation falls back on C and is much faster.
        influence = self.model.get_influence()
        self.cooks_d = influence.cooks_distance[0]

        threshold = 4 / len(self.cooks_d)
        self.influential_outliers_threshold = threshold

        self.influential_points = [i for i, val in enumerate(self.cooks_d) if val > threshold]

    # ------------------------------------------------------------------
    # ASSUMPTION REPORTING (SHARED)
    # ------------------------------------------------------------------   

    def _report_vif(self, vif_threshold=5):
        df = self.vif_results.copy()
        df = df[~df['feature'].str.lower().isin(['constant', 'intercept', 'const'])]
        df = df.sort_values('VIF', ascending=False)
        
        print("\nVariance Inflation Factor (VIF):")
        print(df.to_string(index=False))
        
        problematic = df[df['VIF'] > vif_threshold]
        
        if not problematic.empty:
            print(f"\nVariables with VIF > {vif_threshold}:")
            print(problematic[['feature', 'VIF']].to_string(index=False))
        else:
            print(f"\nNo variables with VIF > {vif_threshold}")

    def _report_influential_outliers(self):
        print(f"Above limit points ({self.influential_outliers_threshold:.3f}): {self.influential_points}")
    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    def _data_cleaning(self):
        self.data = self.data.dropna()
    
    def _preprocessing(self):
        self.y, self.X, self.XList = RapidPreprocessor.prepare_data(
        df=self.data,
        target_cols=[self.outcome_str],
        predictor_cols=self.predictors_list,
        intercept=True
        )
    
    def _modeling(self):
        model = sm.GLM(endog=self.y, exog=self.X, family=self.family)
        self.model = model.fit()

    def _model_evaluation(self, labels, cv = False, splits = 5):
        self._setup_result_summary(labels)
        self._test_performance_metrics()
        if cv:
            self._test_cross_validation(splits)
        self._test_assumptions()

    def _validation():
        pass

    def _visualization():
        pass

    # ------------------------------------------------------------------
    # NECESSARY DATA VALIDATIONS BEFORE PREPROCESSING
    # ------------------------------------------------------------------
    def _run_data_validations(self, data, outcome_str, predictors_list, regression_type):
        self._validate_inputs(data, outcome_str, predictors_list, regression_type)
    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR FOR FIT)
    # ------------------------------------------------------------------
    def _setup_result_summary(self, labels: dict = None):
        """
        Builds all generic parts of the result summary and calls
        abstract methods to build parts specific to different regression types.
        """
        result = self.model
        self.summary_table = result.summary2().tables[1].copy()
        self._build_result_summary_df(labels)
        self.summary_df['Variable'] = self.summary_df['Variable'].str.replace('T.', '')
        for col in self.summary_df.columns[1:-1]:
            self.summary_df[col] = self.summary_df[col].round(3)
        self.summary_df['p-value'] = self.summary_df['p-value'].apply(lambda x: f'{x:.4f}')
        self.summary_df = self.summary_df[self.summary_df['Variable'] != 'Intercept']
        self._rename_cols_by_regression_type()

    def _map_variable_label(self, df: pd.DataFrame, labels: dict = None) -> pd.DataFrame:
        if not labels:
            return df
        
        df = df.copy()
        df['Variable'] = df['Variable'].apply(lambda x: self._parse_variable_name(x, labels))
        return df

    def _parse_variable_name(self, var_name, labels: dict):
        if var_name == 'Intercept':
            return labels.get('Intercept', 'Intercept')
        elif '[' in var_name:
            base_var = var_name.split('[')[0]
            level = var_name.split('[')[1].split(']')[0]
            base_var_name = base_var.replace('C(', '').replace(')', '').strip()
            label = labels.get(base_var_name, base_var_name)
            return f'{label} ({level})'
        else:
            var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
            return labels.get(var_name_clean, var_name_clean)

    @abstractmethod
    def _build_result_summary_df(self):
        """Builds the summary dataframe per linear or logistic regression."""
        pass

    @abstractmethod
    def _rename_cols_by_regression_type(self):
        """
        Renames the result summary dataframe columns from fit per regression type (linear or logistic, 
        as well as univariate or multivariate).
        """
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (MODEL EVALUATION)
    # ------------------------------------------------------------------
    def _test_performance_metrics(self):
        pass

    def _test_assumptions(self):
        pass

    def _test_cross_validation(self):
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (USER INPUT VALIDATION)
    # ------------------------------------------------------------------

    def _validate_inputs(self, data, outcome_str, predictors_list, regression_type):
            # Validate inputs
        if data is None:
            raise ValueError("data cannot be None")
        
        if data.empty:
            raise ValueError("data cannot be empty")
        
        if outcome_str is None or outcome_str == "":
            raise ValueError("outcome_str cannot be None or empty")
        
        if predictors_list is None or len(predictors_list) == 0:
            raise ValueError("predictors_list cannot be None or empty")
        
        if regression_type is None:
            raise ValueError("regression_type cannot be None")
        
        # Check if outcome exists in data
        if outcome_str not in data.columns:
            raise ValueError(f"Outcome variable '{outcome_str}' not found in data columns")
        
        # Check if predictors exist in data
        missing_predictors = [p for p in predictors_list if p not in data.columns]
        if missing_predictors:
            raise ValueError(f"Predictor(s) not found in data columns: {missing_predictors}")
        
    # ------------------------------------------------------------------
    # PROPERTY (STATSMODEL FAMILY)
    # ------------------------------------------------------------------
    @property
    def family():
        pass