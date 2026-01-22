import plotly.io as pio
# This forces Plotly to save an HTML file and try to open it in the browser
#pio.renderers.default = "browser"
import pandas as pd
import warnings
# Import the updated survival class
from survival import RAPID_survival

# Ignore standard runtime warnings during optimization steps
warnings.filterwarnings('ignore', category=RuntimeWarning)


# --- 1. Data Loading ---
try:
    print("--- 1. Loading DataFrames ---")
    # Load model and map datasets from CSV files
    df_model = pd.read_csv('df_model.csv')
    df_map = pd.read_csv('df_map.csv')
    print(f"Datasets loaded successfully. Rows in Case 1: {len(df_model)}.")
except FileNotFoundError:
    print("ERROR: CSV files not found in the execution directory. Check 'df_model.csv' and 'df_map.csv'.")
    exit()

# =================================================================
#                           USER CASE 1
# =================================================================
print("\n" + "="*20 + " STARTING USER CASE 1 " + "="*20)

pipeline_c1 = RAPID_survival(
    data=df_model,
    duration_col='HospitalLengthStay_trunc',
    event_col='HospitalDischargeCode_trunc_bin',
    predictors=[
        'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
        'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
        'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
        'Vent_Resource'
    ]
)

print("\n--- PHASE: DATA PREPARATION (CASE 1) ---")
#This now triggers _data_cleaning() and _preprocessing() internally
pipeline_c1.preprocess_data()

print("\n--- PHASE: MODEL FITTING (CASE 1) ---")
# This now triggers _modeling() and _model_evaluation() internally
pipeline_c1.fit(penalizer=0.1)

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS (CASE 1) ---")
# This now triggers _visualization() internally
pipeline_c1.summary(
    plots=['forest_plot', 'roc_auc'], 
    target_time=40.0
)

print("="*20 + " USER CASE 1 COMPLETE " + "="*20)


# =================================================================
#                           USER CASE 2
# =================================================================
print("\n" + "="*20 + " STARTING USER CASE 2 " + "="*20)

# Manual Pre-processing: Pre-processing specific to Case 2 before passing to the pipeline
df_cox_prep = df_map.copy()
df_cox_prep['dates_admdate'] = pd.to_datetime(df_cox_prep['dates_admdate'], errors='coerce')
df_cox_prep['outco_date'] = pd.to_datetime(df_cox_prep['outco_date'], errors='coerce')
df_cox_prep['duration_col'] = (df_cox_prep['outco_date'] - df_cox_prep['dates_admdate']).dt.days

df_cox_prep['outcome_binary'] = df_cox_prep['outco_binary_outcome'].map({
    "Death": 1, "Censored": 0, "Discharged": 0
})

pipeline_c2 = RAPID_survival(
    data=df_cox_prep,
    duration_col='duration_col',
    event_col='outcome_binary',
    predictors=[
        'demog_sex', 'demog_healthcare',
        'comor_hypertensi', 'comor_chrkidney', 'comor_liverdisease', 'comor_obesity'
    ]
)

print("\n--- PHASE: PRE-PROCESSING (CASE 2) ---")
pipeline_c2.preprocess_data()

print("\n--- PHASE: MODEL FITTING (CASE 2) ---")
pipeline_c2.fit(penalizer=0.1)

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS (CASE 2) ---")
pipeline_c2.summary(
    plots=['forest_plot', 'roc_auc'], 
    target_time=12.0
)


print("\n" + "="*20 + " ALL CASES COMPLETE " + "="*20)