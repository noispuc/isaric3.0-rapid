from regression.py import RAPID_base_regression
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class RAPID_logistic_regression(RAPID_base_regression):
    family = sm.families.Gaussian