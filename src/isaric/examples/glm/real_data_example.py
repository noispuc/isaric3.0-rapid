import pandas as pd
from pathlib import Path
from isaric.modeling.glm import RAPID_GLM

# THIS USE CASE REQUIRES OPENPYXL. Run pip install openpyxl if you do not have this dependency.
data_path = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'dados_uti_ems.xlsx'

# Load the data
df = pd.read_excel(data_path)

# Handle missing values (simple approach - drop rows with any missing values in our selected variables)
selected_vars = ['los', 'Age', 'SofaScore', 'Saps3Points', 'CharlsonComorbidityIndex', 'expected_los']
df_clean = df[selected_vars].dropna()

# Define outcome and independent_vars
dependent_var = 'los'
independent_vars = ['Age', 'SofaScore', 'Saps3Points', 'CharlsonComorbidityIndex', 'expected_los']

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
model = RAPID_GLM(
    data=df_clean,
    dependent_var=dependent_var,
    independent_vars=independent_vars,
    regression_type="Multi"
)

# Fit the model with cross-validation
model.fit(labels=labels, cross_val=True, n_splits=5)

# Display comprehensive summary
model.summary(
    assumptions='all',
    performance='all',
    cross_val='all',
    plots=['forest_plot', 'residuals_vs_fitted', 'qq_plot'],
    vif_threshold=5.0
)

# Save results if needed
# model.summary_df.to_csv('linear_regression_results.csv', index=False)