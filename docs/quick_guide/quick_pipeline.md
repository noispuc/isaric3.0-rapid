
## Quickstart Guide: End-to-End Analysis (Cox PH Example)

This guide demonstrates how to instantiate and run a full survival analysis by defining the problem's scope (Duration, Event, Predictors) and letting the pipeline handle the rest.

### 1.1. Preparing Data Scope

In this phase, you load your data and define the specific columns necessary for the model.

```python
import pandas as pd
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

# Load your prepared dataset (Example: df_model.csv from User Case 1)
try:
    df_model = pd.read_csv('df_model.csv')
except FileNotFoundError:
    print("Please ensure 'df_model.csv' is in your directory.")
    exit()

# Define the necessary columns
DURATION_VAR = 'HospitalLengthStay_trunc'
DEPENDENT_VAR = 'HospitalDischargeCode_trunc_bin'
INDEPENDENT_VARS = [
    'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
    'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
    'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
    'Vent_Resource'
]
TARGET_TIME = 60.0 # Time point (e.g., 60 days) for time-dependent metrics
```

### 1.2. Initializing and Fitting the Model (Phase 1 & 2)

The `.fit()` method calls the internal `.preprocess_data()` to clean and encode variables before training the Cox PH model.

```python
# 1. Initialize the Pipeline (Note: Class name adapted for the tutorial)
cox_pipeline = factory.create(
    "survival",
    data=df_model,
    duration_var=DURATION_COL,
    dependent_var=EVENT_COL,
    independent_vars=PREDICTORS
)

# 2. Fit the Model (Phase 2)
print("\n--- PHASE 2: FITTING THE MODEL ---")
cox_pipeline.fit() 
# Output: Cox PH model fitted successfully on XXXXX observations.
```
In the 'fit' method you can also use data 


### 1.3. Running the Full Summary and Diagnostics (Phase 3)

The `.summary()` method automatically prints the publication-ready table, model fit metrics, and generates all requested plots.

---
> [!IMPORTANT] As many assumptions and validation metrics in the Survival-Cox method are visualized as graphics, they are called in "plots" and not "assumptions"
--- 

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

## 2\. In-Depth Diagnostics and Interpretation

The pipeline uses standardized methods to check the critical assumptions of the Cox PH model and evaluate its performance.

### 2.1. Hazard Ratio (HR) Table

The output table (generated first by `.summary()`) provides the primary interpretative measures.

  * **HR \> 1**: The covariate **increases** the risk of the event (shorter survival).
  * **HR \< 1**: The covariate **decreases** the risk of the event (longer survival).
  


### 2.2. Checking Proportional Hazards (PH) Assumption

The PH assumption requires that the effect of a covariate remains constant over time.

  * **Method:** `_plot_schoenfeld_residuals(covariate_name)` (Called internally by `.summary()`)
  * **Interpretation:** The statistical test output and the plot of Schoenfeld residuals confirm if the assumption holds.


---
> [!IMPORTANT] As many assumptions and validation metrics in the Survival-Cox method are visualized as graphics, they are called in "plots" and not "assumptions"
--- 

### 2.3. Checking Linearity (Martingale/Deviance Residuals)

Martingale and Deviance residuals are used to check the functional form of continuous predictors (linearity) and identify outliers.

  * **Method:** `_plot_martingale_residuals(covariate_name)` and `_plot_deviance_residuals(covariate_name)`
  * **Interpretation:** For categorical variables (using boxplots), distributions should be centered around zero. For continuous variables (using scatter plots with LOESS smoothers), the smoothed line should be flat and near zero.

### 2.4. Discrimination and Calibration

These plots evaluate the predictive utility of the model.

  * **Concordance Index (C-Index):** A global measure of discrimination (ranking ability). It ranges from 0.5 (random chance) to 1.0 (perfect prediction).
  * **Time-Dependent ROC / AUC:** Measures discrimination at a specific time point ($t$).
  * **Calibration Plot:** Compares predicted survival probabilities with observed Kaplan-Meier survival estimates. Points near the 45° diagonal line indicate perfect calibration.