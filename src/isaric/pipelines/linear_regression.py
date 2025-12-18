import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
    # 2: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, plots: list = None):
        """
        Reports the results of the linear regression, generating tables and plots.
        """
        super().summary()
    
    # ------------------------------------------------------------------
    # STATSMODEL FAMILY FOR THIS REGRESSION.
    # ------------------------------------------------------------------
    def family(self):
        return sm.families.Gaussian
    
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
