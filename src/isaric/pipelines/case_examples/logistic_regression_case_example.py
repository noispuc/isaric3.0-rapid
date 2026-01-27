import pandas as pd
from pathlib import Path
import warnings
from isaric.pipelines.logistic_regression import RAPID_LogisticRegression

data_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'df_model.csv'
df = pd.read_csv(data_path)

outcome_str = 'HospitalDischargeCode_trunc_bin'

warnings.filterwarnings('ignore')

# Define predictor variables
predictors_list = [
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
    outcome_str=outcome_str,
    predictors_list=predictors_list,
    regression_type="Multi",
    classification_threshold=0.5
)

# Fit the model
model.fit()

# Display model summary
print(model.model.summary())

# Generate comprehensive summary with diagnostics
model.summary(
    assumptions=True,
    performance=True,
    plots=['forest_plot', 'roc_curve', 'confusion_matrix'],
    cross_val=False,
    vif_threshold=5.0
)

# Access results dataframe
print(model.summary_df)

# Save results if needed
#model.summary_df.to_csv('logistic_regression_results.csv', index=False)