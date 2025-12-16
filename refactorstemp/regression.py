import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class RAPID_base_regression:
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        self.data = data.copy() #This copy instruction ensures that any preprocessing changes only the df in the instance.
        self.outcome_str = outcome_str
        self.predictors_list = predictors_list
        self.regression_type = regression_type
        self.formula = self._build_formula_string(self)

    def preprocess_data(self):
        data = self.data
        predictors_list = self.predictors_list
        #Convert categorical variables to the 'category' type
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.intersection(predictors_list)
        for var in categorical_vars:
            #This loop changes the df inside the instance.
            data[var] = data[var].astype('category')

    def fit(self):
        family = self._get_family()
        model = smf.glm(formula=self.formula, data=self.data, family=family)
        self.model_result = model.fit()

    def summary():
        pass

    def _get_family():
        raise NotImplementedError("Subclasses must define family and family getter function.")

    def _build_formula_string(self):
        self.formula = self.outcome_str + ' ~ ' + ' + '.join(self.predictors_list)
        return
    

    
