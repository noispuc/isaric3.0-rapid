import pandas as pd
import warnings
from isaric.pipelines.logistic_regression import RAPID_LogisticRegression

# Ignore standard runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Data Loading ---
try:
    print("--- 1. Loading Clinical Dataset ---")
    df_logistic = pd.read_csv('df_patient_outcomes.csv')
    print(f"Dataset loaded successfully. Rows: {len(df_logistic)}.")
except FileNotFoundError:
    print("ERROR: CSV file not found.")
    exit()

# =================================================================
#                USE CASE: PREDICTING PATIENT MORTALITY
# =================================================================
print("\n" + "="*20 + " STARTING MORTALITY ANALYSIS " + "="*20)

# Initialize the Logistic Regression Pipeline
# Outcome: death_bin (1 = Deceased, 0 = Survived)
pipeline_log = RAPID_LogisticRegression(
    data=df_logistic,
    outcome_str='death_bin',
    predictors_list=[
        'age', 'sex', 'obesity',         # Demographics
        'comor_diabetes', 'comor_copd',  # Chronic conditions
        'is_immunossupression'           # Risk factors
    ],
    regression_type="Multi",
    classification_threshold=0.5         # Threshold for the Confusion Matrix
)

print("\n--- PHASE: MODEL FITTING & CROSS-VALIDATION ---")
# This triggers _validate_binary_outcome internally to ensure outcome is 0/1
pipeline_log.fit(
    cross_val=True, 
    n_splits=5,
    labels={
        'death_bin': 'Mortality Status',
        'comor_diabetes': 'Diabetes Mellitus',
        'is_immunossupression': 'Immunosuppressed'
    }
)

print("\n--- PHASE: SUMMARY AND DIAGNOSTICS ---")
# Generate the clinical report including classification metrics
pipeline_log.summary(
    assumptions=True,   # Shows EPV, VIF, and Influential Outliers (Cook's D)
    performance=True,   # Shows Accuracy, F1-Score, Precision, Recall, and Log Loss
    cross_val=True,     # Shows Mean Accuracy across 5 folds
    plots=[
        'forest_plot',      # Visualize Odds Ratios (log scale)
        'roc_curve',        # Area Under the Curve (AUC)
        'confusion_matrix'  # True Positives vs False Positives
    ]
)

print("\n" + "="*20 + " ANALYSIS COMPLETE " + "="*20)