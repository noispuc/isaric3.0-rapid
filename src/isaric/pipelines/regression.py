from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class RAPID_base_regression:
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        self.data = data
        self.outcome_str = outcome_str
        self.predictors_list = predictors_list
        self.regression_type = regression_type
        self._build_formula_string()

    def preprocess_data(self):
        data = self.data
        predictors_list = self.predictors_list
        #Convert categorical variables to the 'category' type
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.intersection(predictors_list)
        for var in categorical_vars:
            #This loop changes the df inside the instance.
            data[var] = data[var].astype('category')

    #Method to fit model.
    def fit(self, labels: dict = None):
        model = smf.glm(formula=self.formula, data=self.data, family=self.family)
        self.model_result = model.fit()

    #Abstract property that determines statsmodel family for each regression.
    @property
    @abstractmethod
    def family(self):
        """Statsmodels family used by this regression."""
        pass

    #Abstract method to provide the summary and graphics.
    @abstractmethod
    def summary(self):
        """Summary to be output by this regression."""
        pass


    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR)
    # ------------------------------------------------------------------

    def _setup_result_summary(self, labels : dict = None):
        """
        Builds all generic parts of the result summary and calls
        abstract methods to build parts specific to different regression types.
        """
        result = self.model_result
        self.summary_table = result.summary2().tables[1].copy()

        self._build_result_summary_df(labels)

        self.summary_df['Study'] = self.summary_df['Study'].str.replace('T.', '')

        for col in self.summary_df.columns[1:-1]:
            self.summary_df[col] = self.summary_df[col].round(3)

        self.summary_df['p-value'] = self.summary_df['p-value'].apply(lambda x: f'{x:.4f}')
        self.summary_df = self.summary_df[self.summary_df['Study'] != 'Intercept']
        self._rename_cols_by_regression_type()

    def _map_study_label(self, df: pd.DataFrame, labels : dict = None) -> pd.DataFrame:
        if not labels:
            return df
        
        df = df.copy()
        df['Study'] = df['Study'].apply(self._parse_variable_name)
        return df

    def _parse_variable_name(self, var_name, labels : dict):
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
    # PRIVATE METHOD (FORMULA STRING BUILDER)
    # ------------------------------------------------------------------

    def _build_formula_string(self):
        self.formula = self.outcome_str + ' ~ ' + ' + '.join(self.predictors_list)
        return
    

    
