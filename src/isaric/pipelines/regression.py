from abc import abstractmethod

import pandas as pd
import statsmodels.api as sm
import numpy as np

from isaric.pipelines.modules.rapid_preprocess import RapidPreprocessor
from isaric.pipelines.modules.rapid_assumption import ModelAssumptionTester
from .pipeline import RAPID_BasePipeline



class RAPID_BaseRegression(RAPID_BasePipeline):
    def __init__(self, data: pd.DataFrame, yvar: str = None, predictors: list = None, formula: str = None, 
                family: str = None, link:str = None, regression_type: str = "Multi"):
        self._run_data_validations(data, yvar, predictors, formula, family, link, regression_type)
        self.data = data.copy()
        self.yvar = yvar
        self.predictors = predictors
        self.formula = formula

        family_cls = self._family_map[family.lower()]
        link_obj = self._link_map[link.lower()]()
        self.family = family_cls(link=link_obj)

        self.regression_type = regression_type
        self.performance_metrics_df = None
        self.assumption_metrics_df = None
        self._preprocess_data()
    
    # ------------------------------------------------------------------
    # DATA PREPROCESSING
    # ------------------------------------------------------------------

    def _preprocess_data(self):
        self._data_cleaning()
        self._preprocessing()


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
        if (self.X is None or self.y is None):
            print("Please run preprocess_data before fitting model.")
        self._modeling()
        self._model_evaluation(labels=labels, cv=cross_val, splits=n_splits)

    def summary(self):
        """
        Summary to be output by this regression
        """
        if self.fitted_model is None:
            print("self.model does not exist. Fit the model before calling summary.")
            return

    # ------------------------------------------------------------------
    # ASSUMPTION TESTING (SHARED)
    # ------------------------------------------------------------------        

    def _setup_assumption_tester(self):
        self.assumption_tester = ModelAssumptionTester(model=self.fitted_model, X=self.X, y=self.y, y_pred=self.fitted_model.fittedvalues)

    def _evaluate_vif(self):
        self.vif_df = self.assumption_tester.test_vif()
        
    def _evaluate_influential_outliers(self):
        """
        Evaluates influential outliers using Cook's distance.
        This is shared between linear and logistic regression.
        """
        influence = self.fitted_model.get_influence()
        self.cooks_d = influence.cooks_distance[0]

        threshold = 4 / len(self.cooks_d)
        self.influential_outliers_threshold = threshold

        self.influential_points = [i for i, val in enumerate(self.cooks_d) if val > threshold]

    # ------------------------------------------------------------------
    # ASSUMPTION REPORTING (SHARED)
    # ------------------------------------------------------------------   

    @abstractmethod
    def _build_assumption_metrics_df(self, vif_threshold: float = 5.0):
        """
        Build assumption metrics dataframe.
        Each regression type implements its own specific assumptions.
        """
        pass

    def _report_assumptions(self, vif_threshold: float = 5.0, metrics=None):
        if self.assumption_metrics_df is None:
            self._build_assumption_metrics_df(vif_threshold)

        df = self.assumption_metrics_df
        if metrics != 'all':
            missing = set(metrics) - set(df['Test']) - {'VIF', 'Influential Outliers'}
            if missing:
                print(f"Warning: the following assumption metrics were not found: {missing}")
            df = df[df['Test'].isin(metrics)]

        print("=" * 80)
        print("ASSUMPTION TEST RESULTS")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        if metrics == 'all' or 'VIF' in metrics:
            self._report_vif_table(vif_threshold)

        if metrics == 'all' or 'Influential Outliers' in metrics:
            self._report_influential_outliers_details()
    
    def _report_vif_table(self, vif_threshold: float = 5.0):
        if hasattr(self, 'vif_df') and self.vif_df is not None:
            print("\n")
            print("=" * 80)
            print("VARIANCE INFLATION FACTOR (VIF) - MULTICOLLINEARITY CHECK")
            print("=" * 80)
            vif_display = self.vif_df.copy()
            vif_display = vif_display[~vif_display['feature'].str.lower().isin(['intercept', 'const', 'constant'])]
            vif_display['Interpretation'] = vif_display['VIF'].apply(
                lambda x: 'High multicollinearity' if x > vif_threshold else 'Acceptable'
            )
            print(vif_display.to_string(index=False))
            print("=" * 80)
            print(f"Note: VIF > {vif_threshold} indicates potential multicollinearity issues")
            print("=" * 80)
    
    def _report_influential_outliers_details(self):
        if hasattr(self, 'influential_points') and len(self.influential_points) > 0:
            print("\n")
            print("=" * 80)
            print("INFLUENTIAL OUTLIERS DETAILS")
            print("=" * 80)
            print(f"Threshold (4/n): {self.influential_outliers_threshold:.6f}")
            print(f"Number of influential points: {len(self.influential_points)}")
            print(f"Influential point indices: {self.influential_points}")
            print("=" * 80)
    
    # ------------------------------------------------------------------
    # PERFORMANCE METRICS (SHARED)
    # ------------------------------------------------------------------
    def _evaluate_aic_bic(self):
        self.aic = self.fitted_model.aic
        self.bic = self.fitted_model.bic
        self.llf = self.fitted_model.llf

    def _evaluate_glm_r2(self):
        k = int(self.fitted_model.df_model) + 1
        ll_model = self.fitted_model.llf
        ll_null = self.fitted_model.llnull

        self.mcfadden_r2 = 1 - (ll_model / ll_null)
        self.mcfadden_adj_r2 = 1 - ((ll_model - k) / ll_null)

        y_array = np.asarray(self.y).ravel()
        fitted = np.asarray(self.fitted_model.fittedvalues).ravel()
        y_mean = y_array.mean()
        self.efron_r2 = 1 - (np.sum((y_array - fitted) ** 2) / np.sum((y_array - y_mean) ** 2))

    # ------------------------------------------------------------------
    # PERFORMANCE METRICS (ABSTRACT - DIFFERENT FOR EACH REGRESSION)
    # ------------------------------------------------------------------
    
    @abstractmethod
    def _build_performance_metrics_df(self):
        """
        Build performance metrics dataframe.
        Each regression type has different metrics.
        """
        pass
    
    @abstractmethod
    def _report_performance(self):
        """
        Report performance metrics.
        Each regression type implements its own reporting.
        """
        pass

    # ------------------------------------------------------------------
    # CROSS VALIDATION DATAFRAME (ABSTRACT - DIFFERNET FOR EACH REGRESSION)
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_cv_df(self):
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------

    def _data_cleaning(self):
        self.data = self.data.dropna()
    
    def _preprocessing(self):
        self.y, self.X, self.XList = RapidPreprocessor.prepare_data(
        df=self.data,
        formula=self.formula,
        target_cols=[self.yvar],
        predictor_cols=self.predictors,
        intercept=True
        )
    
    def _modeling(self):
        model = sm.GLM(endog=self.y, exog=self.X, family=self.family)
        self.fitted_model = model.fit()
        

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
    def _run_data_validations(self, data, yvar, predictors, formula, family, link, regression_type):
        self._validate_inputs(data, yvar, predictors, formula, family, link, regression_type)
    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR FOR FIT)
    # ------------------------------------------------------------------
    def _setup_result_summary(self, labels: dict = None):
        """
        Builds all generic parts of the result summary and calls
        abstract methods to build parts specific to different regression types.
        """
        result = self.fitted_model
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
        self._evaluate_glm_r2()
        self._evaluate_aic_bic()

    def _test_assumptions(self):
        pass

    def _test_cross_validation(self):
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (USER INPUT VALIDATION)
    # ------------------------------------------------------------------

    def _validate_inputs(self, data, yvar, predictors, formula, family, link, regression_type):
            # Validate inputs
        if data is None:
            raise ValueError("data cannot be None")
        
        if data.empty:
            raise ValueError("data cannot be empty")
        
        if family is None or family not in self._family_map.keys():
            raise ValueError(f"family cannot be empty and must be: {self._family_map.keys()}")
        
        if link is None or link not in self._link_map.keys():
            raise ValueError(f"link cannot be empty and must be: {self._link_map.keys()}")
        
        if (yvar is None or yvar == "") and formula == None:
            raise ValueError("yvar cannot be None or empty if a formula is not provided.")
        
        if (predictors is None or len(predictors) == 0) and formula == None:
            raise ValueError("predictors cannot be None or empty if a formula is not provided.")
        
        if regression_type is None:
            raise ValueError("regression_type cannot be None")
        
        # Check if outcome exists in data
        if (yvar):
            if yvar not in data.columns:
                raise ValueError(f"Outcome variable '{yvar}' not found in data columns")
        
        # Check if predictors exist in data
        if (predictors):
            missing_predictors = [p for p in predictors if p not in data.columns]
            if missing_predictors:
                raise ValueError(f"Predictor(s) not found in data columns: {missing_predictors}")
        
    # ------------------------------------------------------------------
    # DICTIONARY MAPPINGS FOR SM FAMILY AND LINK
    # ------------------------------------------------------------------
    @property
    def _family_map(self):
        pass
    @property
    def _link_map(self):
        pass