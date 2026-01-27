import pandas as pd
import warnings
from isaric.pipelines.linear_regression import RAPID_LinearRegression

# Ignore standard runtime warnings during optimization/matrix inversion
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Data Loading ---
try:
    print("--- 1. Loading Clinical Dataset ---")
    # Assuming the dataset contains the features used in the notebook
    df_clinical = pd.read_csv('df_clinical_data.csv')
    print(f"Dataset loaded successfully. Rows: {len(df_clinical)}.")
except FileNotFoundError:
    print("ERROR: Clinical CSV file not found.")
    exit()

# =================================================================
#                USE CASE: PREDICTING RESPIRATORY RATE
# =================================================================
print("\n" + "="*20 + " STARTING RESPIRATORY ANALYSIS " + "="*20)

# Initialize the Linear Regression Pipeline
# Outcome: vital_rr (Respiratory Rate)
pipeline_lr = RAPID_LinearRegression(
    data=df_clinical,
    outcome_str='vital_rr',
    predictors_list=[
        'age', 'sex',                     # Demographics
        'comor_hypertension', 'comor_dm', # Comorbidities (Hypertension/Diabetes)
        'lab_temp_max', 'lab_creatinine'  # Clinical Labs
    ],
    regression_type="Multi"
)

print("\n--- PHASE: MODEL FITTING & CROSS-VALIDATION ---")
# fits model, calculates VIF, Cook's Distance, and performs 5-fold CV
pipeline_lr.fit(
    cross_val=True, 
    n_splits=5,
    labels={
        'vital_rr': 'Respiratory Rate (bpm)',
        'lab_temp_max': 'Max Temperature (°C)',
        'comor_dm': 'Diabetes Mellitus'
    }
)

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS ---")
# Generate a full report including statistical assumptions and performance
pipeline_lr.summary(
    assumptions=True,   # Shows Durbin-Watson, Shapiro-Wilk, and VIF
    performance=True,   # Shows MSE, RMSE, and R-Squared
    cross_val=True,     # Shows the stability of MSE across folds
    plots=[
        'forest_plot',          # Visualize Coefficient Estimates
        'residuals_vs_fitted',  # Check for Homoscedasticity
        'qq_plot'               # Check for Normality of Errors
    ],
    vif_threshold=5.0
)

print("\n" + "="*20 + " ANALYSIS COMPLETE " + "="*20)