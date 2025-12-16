import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from regression import RAPID_base_regression


class RAPID_linear_regression(RAPID_base_regression):
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        super().__init__(self,data,outcome_str,predictors_list,regression_type)
    
    def fit(self, labels: dict = None):
        """
        Fits the model, (in this case a linear regression model) using the pre-specified data and predictors.
        Stores the fitted model and summary results internally.
        """
        super().fit(labels)
        self._setup_result_summary(labels)


    def summary(self):
        """
        Reports the results of the linear regression, generating publication-ready tables and plots.

        Args:
            temp
        """
        pass

    def family(self):
        return sm.families.Gaussian
    
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
    
    