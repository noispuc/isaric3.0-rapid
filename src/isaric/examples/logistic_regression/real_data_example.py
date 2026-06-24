import pandas as pd
from pathlib import Path
import warnings
from isaric.modeling.logistic_regression import RAPID_LogisticRegression

data_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'df_model.csv'
df = pd.read_csv(data_path)
warnings.filterwarnings('ignore')

# Define outcome and predictor variables
dependent_var = 'HospitalDischargeCode_trunc_bin'
independent_vars = [
    'Age',
    'Gender',
    'Saps3Points',
    'SofaScore',
    'hypertension',
    'diabetes',
    'obesity',
    'cancer',
    'IsVasopressors',
    'ResourceIsMechanicalVentilation',
    'UnitLengthStay'
]

# Initialize the logistic regression model
model = RAPID_LogisticRegression(
    data=df,
    dependent_var=dependent_var,
    independent_vars=independent_vars,
    regression_type="Multi",
    classification_threshold=0.5
)

# Fit the model
model.fit()

# Display statsmodels summary
print(model.fitted_model.summary())

# Generate comprehensive summary with diagnostics
model.summary(
    assumptions='all',
    performance='all',
    plots=['forest_plot', 'roc_curve', 'confusion_matrix'],
    cross_val=None,
    vif_threshold=5.0
)

# Access results dataframe
print(model.summary_df)

# Save results if needed
# model.summary_df.to_csv('logistic_regression_results.csv', index=False)