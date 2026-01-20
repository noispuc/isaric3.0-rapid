import plotly.io as pio

# This forces Plotly to save an HTML file and try to open it in the browser
#pio.renderers.default = "browser"
import pandas as pd
import warnings
# Import the updated survival class
from pipelines/modules/rapid_plots.py import RAPID_survival

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

# Define time-to-event and status columns
duration_col_c1 = 'HospitalLengthStay_trunc'
event_col_c1 = 'HospitalDischargeCode_trunc_bin'

# Define predictors for the first scenario
predictors_c1 = [
    'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
    'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
    'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
    'Vent_Resource'
]

# Set evaluation time point (e.g., 60-day survival)
target_time_c1 = 60.0

# Initialize the pipeline for Case 1
pipeline_c1 = RAPID_survival(
    data=df_model,
    duration_col=duration_col_c1,
    event_col=event_col_c1,
    predictors=predictors_c1
)

print("\n--- PHASE: MODEL FITTING (CASE 1) ---")
pipeline_c1.fit()

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS (CASE 1) ---")
pipeline_c1.summary(
    plots=['forest_plot', 'roc_auc'], 
    target_time=target_time_c1
)

print("="*20 + " USER CASE 1 COMPLETE " + "="*20)


# =================================================================
#                           USER CASE 2
# =================================================================
print("\n" + "="*20 + " STARTING USER CASE 2 " + "="*20)

# Manual Pre-processing: Calculate duration from admission and outcome dates
df_cox_prep = df_map.copy()
df_cox_prep['dates_admdate'] = pd.to_datetime(df_cox_prep['dates_admdate'], errors='coerce')
df_cox_prep['outco_date'] = pd.to_datetime(df_cox_prep['outco_date'], errors='coerce')

# Calculate duration in days
df_cox_prep['duration_col'] = (df_cox_prep['outco_date'] - df_cox_prep['dates_admdate']).dt.days

# 2. Map Categorical Outcome to Binary
# lifelines requires 1 for event and 0 for censorship
df_cox_prep['outcome_binary'] = df_cox_prep['outco_binary_outcome'].map({
    "Death": 1, 
    "Censored": 0, 
    "Discharged": 0
})

# 3. Setup Predictors and Target Time
duration_col_c2 = 'duration_col'
event_col_c2 = 'outcome_binary'
predictors_c2 = [
    'demog_sex', 'demog_healthcare',
    'comor_hypertensi', 'comor_chrkidney', 'comor_liverdisease', 'comor_obesity', 
    'comor_chrkidney_stag', 'comor_liverdisease_type'
]
target_time_c2 = 12.0 # Evaluation point for Case 2

# 4. Initialize and Run Pipeline
pipeline_c2 = RAPID_survival(
    data=df_cox_prep,
    duration_col=duration_col_c2,
    event_col=event_col_c2,
    predictors=predictors_c2
)

print("\n--- PHASE: MODEL FITTING (CASE 2) ---")
pipeline_c2.fit()

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS (CASE 2) ---")
# Use the plots list to trigger visualizations
pipeline_c2.summary(
    plots=['forest_plot', 'roc_auc'], 
    target_time=target_time_c2
)

print("="*20 + " USER CASE 2 COMPLETE " + "="*20)