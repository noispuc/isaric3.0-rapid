import pandas as pd
from pathlib import Path
from isaric.pipelines.linear_regression import RAPID_LinearRegression

#THIS USE CASE REQUIRES OPENPYXL. Run pip install openpyxl if you do not have this dependency.

data_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'dados_uti_ems.xlsx'

# Load the data
df = pd.read_excel(data_path)

# Handle missing values (simple approach - drop rows with any missing values in our selected variables)
selected_vars = ['los', 'Age', 'SofaScore', 'Saps3Points', 'CharlsonComorbidityIndex', 'expected_los']
df_clean = df[selected_vars].dropna()

# Define outcome and predictors
outcome_str = 'los'
predictors_list = ['Age', 'SofaScore', 'Saps3Points', 'CharlsonComorbidityIndex', 'expected_los']

# Optional: Create labels for better visualization
labels = {
    'los': 'Length of Stay (days)',
    'Age': 'Age (years)',
    'SofaScore': 'SOFA Score',
    'Saps3Points': 'SAPS III Points',
    'CharlsonComorbidityIndex': 'Charlson Comorbidity Index',
    'expected_los': 'Expected Length of Stay'
}

# Initialize the pipeline
model = RAPID_LinearRegression(
    data=df_clean,
    outcome_str=outcome_str,
    predictors_list=predictors_list,
    regression_type="Multi"
)

# Fit the model with cross-validation
model.fit(labels=labels, cross_val=True, n_splits=5)

# Display comprehensive summary
model.summary(
    assumptions=True,
    performance=True,
    cross_val=True,
    plots=['forest_plot', 'residuals_vs_fitted', 'qq_plot'],
    vif_threshold=5.0
)

# Save results if needed
#model.summary_df.to_csv('logistic_regression_results.csv', index=False)