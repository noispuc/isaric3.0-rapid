import pandas as pd
import warnings
import numpy as np
from pathlib import Path
from isaric.modeling.survival import RAPID_SurvivalCox

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
# USER CASE 1: Full Performance & Cross-Validation Test
# =================================================================
print("\n" + "="*20 + " TESTING CASE 1: ADVANCED EVALUATION " + "="*20)

# 1. Instantiation
pipeline_c1 = RAPID_SurvivalCox(
    data=df_model,
    duration_var='HospitalLengthStay_trunc',
    dependent_var='HospitalDischargeCode_trunc_bin',
    independent_vars=[
        'period', 'Idade_Agrupada2', 'obesity', 'IsSteroidsUse', 'cancer'
    ]
)

# 2. Fit with Cross-Validation enabled
# Testing the new 'cross_val' and 'n_splits' parameters added to the class
print("Fitting model with 5-fold Cross-Validation...")
pipeline_c1.fit(
    labels={'obesity': 'Clinical Obesity', 'cancer': 'Malignancy'},
    penalizer=0.1,
    cross_val=True, 
    n_splits=5
)

# 3. Summary including All New Plots
# testing: forest_plot, roc_auc, brier_score, and cross-validation reporting
print("Generating comprehensive summary and diagnostic plots...")
pipeline_c1.summary(
    performance=True,
    assumptions=True,
    plots=['forest_plot', 'roc_auc', 'brier_score'], 
    target_time=30.0  # Evaluating at 30 days
)

# =================================================================
# USER CASE 2: Residual Diagnostics & Calibration
# =================================================================
print("\n" + "="*20 + " TESTING CASE 2: RESIDUALS & CALIBRATION " + "="*20)

# 1. Data Prep for Case 2
df_cox_prep = df_map.copy()
df_cox_prep['duration_var'] = (pd.to_datetime(df_cox_prep['outco_date']) - 
                               pd.to_datetime(df_cox_prep['dates_admdate'])).dt.days
df_cox_prep['outcome_binary'] = df_cox_prep['outco_binary_outcome'].map(
    {"Death": 1, "Censored": 0, "Discharged": 0}
)

pipeline_c2 = RAPID_SurvivalCox(
    data=df_cox_prep,
    duration_var='duration_var',
    dependent_var='outcome_binary',
    independent_vars=['demog_sex', 'comor_hypertensi', 'comor_obesity']
)

# 2. Fit using Formula for interaction terms
custom_formula = "duration_var + outcome_binary ~ demog_sex * comor_obesity + comor_hypertensi"
pipeline_c2.fit(formula=custom_formula, penalizer=0.05)

# 3. Comprehensive Residual Testing
# Testing: martingale, deviance, and the new calibration plot method
print("Testing Martingale and Deviance residuals for model fit diagnostics...")
pipeline_c2.summary(
    performance=True,
    assumptions=True,
    plots=['martingale', 'deviance']
)

# 4. Explicitly testing the new Plotly Calibration method
# This calls the internal method directly to ensure Plotly engine is working
print("Rendering Survival Calibration Plot...")
pipeline_c2._render_calibration_plotly(target_time=14.0)

print("\n" + "="*20 + " ALL PIPELINE FEATURES TESTED " + "="*20)