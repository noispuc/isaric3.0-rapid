import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from regression import RAPID_base_regression


class RAPID_logistic_regression(RAPID_base_regression):
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        super().__init__(self,data,outcome_str,predictors_list,regression_type)
    
    def fit(self):
        """
        Fits the model, (in this case a logistic regression model) using the pre-specified data and predictors.
        Stores the fitted model and summary results internally.
        """
        super().fit()
        self._setup_result_summary()

    def summary(self):
        """
        Reports the results of the logistic regression, generating publication-ready tables and plots.

        Args:
            temp
        """
        pass
    
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

    def _build_result_summary_df(self):
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
        self.summary_df = self._map_study_label(self.summary_df)
        self.summary_df = self.summary_df[['Study', 'OddsRatio', 'LowerCI', 'UpperCI', 'p-value']]

    def family(self):
        return sm.families.Binomial
