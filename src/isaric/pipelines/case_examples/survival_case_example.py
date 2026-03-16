# This forces Plotly to save an HTML file and try to open it in the browser
#pio.renderers.default = "browser"
import pandas as pd
from pathlib import Path
import warnings
from isaric.pipelines.survival_cox import RAPID_SurvivalCox

data_path_df_model = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'df_model.csv'
data_path_df_map = Path(__file__).parent.parent.parent.parent.parent / 'data' / 'df_map.csv'

# Ignore standard runtime warnings during optimization steps
warnings.filterwarnings('ignore', category=RuntimeWarning)


# --- 1. Data Loading ---   
try:
    print("--- 1. Loading DataFrames ---")
    # Load model and map datasets from CSV files
    df_model = pd.read_csv(data_path_df_model)
    df_map = pd.read_csv(data_path_df_map)
    print(f"Datasets loaded successfully. Rows in Case 1: {len(df_model)}.")
except FileNotFoundError:
    print("ERROR: CSV files not found in the execution directory. Check 'df_model.csv' and 'df_map.csv'.")
    exit()

# =================================================================
#                           USER CASE 1
# =================================================================
print("\n" + "="*20 + " STARTING USER CASE 1 " + "="*20)

# 1. Instantiation
# independent_vars_list is handled internally by the class
pipeline_c1 = RAPID_SurvivalCox(
    data=df_model,
    duration_var='HospitalLengthStay_trunc',
    dependent_var='HospitalDischargeCode_trunc_bin',
    independent_vars=[
        'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
        'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
        'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
        'Vent_Resource'
    ]
)
# 2. Fit
# Standard labels for clean reporting
labels_c1 = {
    'period': 'Period',
    'Idade_Agrupada2': 'Age Group',
    'obesity': 'Obesity'
}

print("Fitting Model Case 1...")
# Note: Preprocessing (cleaning and matrix generation) happens inside fit()
pipeline_c1.fit(labels=labels_c1, penalizer=0.1, cross_val=True, n_splits=5)

# 3. Summary
# Using boolean flags for performance and assumptions as standardized
pipeline_c1.summary(
    performance=True,
    assumptions=True,
    plots=['forest_plot', 'roc_auc'], 
    target_time=40.0
)

print("="*20 + " USER CASE 1 COMPLETE " + "="*20)


# =================================================================
#                           USER CASE 2
# =================================================================
print("\n" + "="*20 + " STARTING USER CASE 2 (df_map processing) " + "="*20)

# 1. Specific data preparation for df_map
df_cox_prep = df_map.copy()
# Converting dates and calculating duration
df_cox_prep['duration_var'] = (pd.to_datetime(df_cox_prep['outco_date']) - 
                               pd.to_datetime(df_cox_prep['dates_admdate'])).dt.days
# Mapping outcomes to binary
df_cox_prep['outcome_binary'] = df_cox_prep['outco_binary_outcome'].map(
    {"Death": 1, "Censored": 0, "Discharged": 0}
)

# 2. Instantiation with processed df_map
pipeline_c2 = RAPID_SurvivalCox(
    data=df_cox_prep,
    duration_var='duration_var',
    dependent_var='outcome_binary',
    independent_vars=['demog_sex', 'comor_hypertensi', 'comor_obesity']
)

# 3. Fit using a Custom Formula
# This allows testing interactions like Sex * Obesity
custom_formula = "duration_var + outcome_binary ~ demog_sex * comor_obesity + comor_hypertensi"

print("Fitting Model Case 2 with Formula...")
pipeline_c2.fit(formula=custom_formula, penalizer=0.1)

# 4. Summary with Martingale Residuals
# Useful for checking linearity of continuous independent_vars
pipeline_c2.summary(
    performance=True,
    assumptions=True,
    plots=['martingale']
)

print("\n" + "="*20 + " ALL CASES COMPLETE " + "="*20)