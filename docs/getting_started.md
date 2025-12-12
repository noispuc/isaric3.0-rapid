# 🚀 Getting Started with `RAPID_pipeline`

The `RAPID_pipeline` class is designed as a **modular and extensible structure** for implementing various Data Science models (Statistical, Survival Analysis, Machine Learning, etc.).

The current implementation focuses on the **Cox Proportional Hazards (Cox PH) model** for survival analysis, providing automated preprocessing, model fitting, and a comprehensive suite of diagnostic outputs.

Its core goal is to automate data preprocessing, model fitting, and the generation of publication-ready tables and diagnostic plots in a single, reusable object.

## 1\. Core Principles and Modularity

The pipeline is built on a simple three-phase methodology, which is universal for all models you integrate, ensuring high modularity and extensibility:

| Phase | Method | Role in the Pipeline |
| :--- | :--- | :--- |
| **Phase 1: Preprocessing** | `.preprocess_data()` | Handles data cleaning, missing value removal, and feature transformation (e.g., one-hot encoding). |
| **Phase 2: Training** | `.fit()` | Selects the model type (e.g., Cox PH, Logistic Regression, Random Forest) and performs the training procedure. |
| **Phase 3: Output/Diagnostics** | `.summary()` | Generates performance metrics, fit measures (e.g., AIC, C-Index), and produces a user-specified list of diagnostic plots. |

## 2\. Installation (Mock)

To use the current **Survival Analysis** module, ensure the necessary Python packages are installed, all the required packages are listeded in [requirements.txt](https://github.com/noispuc/isaric3.0-rapid/tree/main):

```bash
# Install the core dependencies
pip install pandas numpy matplotlib seaborn statsmodels
# Install the specialized survival analysis library
pip install lifelines scikit-learn plotly
```

## 3\. Quickstart Guide: End-to-End Analysis (Cox PH Example)

This guide demonstrates how to instantiate and run a full survival analysis by defining the problem's scope (Duration, Event, Predictors) and letting the pipeline handle the rest.

### 3.1. Preparing Data Scope

In this phase, you load your data and define the specific columns necessary for the model.

```python
import pandas as pd
# Assuming the file is saved as rapid_pipeline.py
from rapid_pipeline import RAPID_pipeline 

# Load your prepared dataset (Example: df_model.csv from User Case 1)
try:
    df_model = pd.read_csv('df_model.csv')
except FileNotFoundError:
    print("Please ensure 'df_model.csv' is in your directory.")
    exit()

# Define the necessary columns
DURATION_COL = 'HospitalLengthStay_trunc'
EVENT_COL = 'HospitalDischargeCode_trunc_bin'
PREDICTORS = [
    'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
    'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
    'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
    'Vent_Resource'
]
TARGET_TIME = 60.0 # Time point (e.g., 60 days) for time-dependent metrics
```

### 3.2. Initializing and Fitting the Model (Phase 1 & 2)

The `.fit()` method calls the internal `.preprocess_data()` to clean and encode variables before training the Cox PH model.

```python
# 1. Initialize the Pipeline (Note: Class name adapted for the tutorial)
cox_pipeline = RAPID_pipeline(
    data=df_model,
    duration_col=DURATION_COL,
    event_col=EVENT_COL,
    predictors=PREDICTORS
)

# 2. Fit the Model (Phase 2)
print("\n--- PHASE 2: FITTING THE MODEL ---")
cox_pipeline.fit() 
# Output: Cox PH model fitted successfully on XXXXX observations.
```

### 3.3. Running the Full Summary and Diagnostics (Phase 3)

The `.summary()` method automatically prints the publication-ready table, model fit metrics, and generates all requested plots.

```python
# List the plots you want to generate
PLOTS_TO_GENERATE = [
    'forest_plot', 
    'schoenfeld_residuals', # Checks PH assumption
    'martingale_residuals', # Checks functional form (linearity)
    'roc_auc',              # Checks discrimination
    'calibration_plot'      # Checks calibration
]

# 3. Generate Summary and Diagnostics (Phase 3)
print("\n--- PHASE 3: SUMMARY AND DIAGNOSTICS ---")
cox_pipeline.summary(
    fit_measures=True,       # Show AIC/BIC and C-Index
    plots=PLOTS_TO_GENERATE, # Generate the requested plots
    target_time=TARGET_TIME  # Required for ROC/Calibration
)
```

## 4\. In-Depth Diagnostics and Interpretation

The pipeline uses standardized methods to check the critical assumptions of the Cox PH model and evaluate its performance.

### 4.1. Hazard Ratio (HR) Table

The output table (generated first by `.summary()`) provides the primary interpretative measures.

  * **HR \> 1**: The covariate **increases** the risk of the event (shorter survival).
  * **HR \< 1**: The covariate **decreases** the risk of the event (longer survival).

### 4.2. Checking Proportional Hazards (PH) Assumption

The PH assumption requires that the effect of a covariate remains constant over time.

  * **Method:** `_plot_schoenfeld_residuals(covariate_name)` (Called internally by `.summary()`)
  * **Interpretation:** The statistical test output and the plot of Schoenfeld residuals confirm if the assumption holds.

### 4.3. Checking Linearity (Martingale/Deviance Residuals)

Martingale and Deviance residuals are used to check the functional form of continuous predictors (linearity) and identify outliers.

  * **Method:** `_plot_martingale_residuals(covariate_name)` and `_plot_deviance_residuals(covariate_name)`
  * **Interpretation:** For categorical variables (using boxplots), distributions should be centered around zero. For continuous variables (using scatter plots with LOESS smoothers), the smoothed line should be flat and near zero.

### 4.4. Discrimination and Calibration

These plots evaluate the predictive utility of the model.

  * **Concordance Index (C-Index):** A global measure of discrimination (ranking ability). It ranges from 0.5 (random chance) to 1.0 (perfect prediction).
  * **Time-Dependent ROC / AUC:** Measures discrimination at a specific time point ($t$).
  * **Calibration Plot:** Compares predicted survival probabilities with observed Kaplan-Meier survival estimates. Points near the 45° diagonal line indicate perfect calibration.