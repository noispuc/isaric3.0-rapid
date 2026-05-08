
#  Quickstart: Running the RAPID Pipeline

This guide demonstrates how to run an end-to-end analysis using the **RAPID Pipeline**. The same three-phase pattern works for **all models** (Logistic, GLM, Survival, MICE).

---

## 1. The Three-Phase Pattern

| Phase | Method | What it Does |
|:-----:|--------|--------------|
| 1 | `create()` | Initialize model and define data scope |
| 2 | `fit()` | Train model with preprocessing |
| 3 | `summary()` | Generate diagnostics, metrics, and plots |

This pattern is **universal** across all RAPID models.

---

## 2. Example: Survival Analysis (Cox PH)

### 2.1 Prepare Data Scope

```python
import pandas as pd
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

# Load your dataset
df = pd.read_csv('your_data.csv')

# Define the analysis scope
DURATION_VAR = 'HospitalLengthStay'
EVENT_VAR = 'DischargeStatus'  # 1 = event, 0 = censored
PREDICTORS = ['age', 'sex', 'bmi', 'comorbidity_index']
TARGET_TIME = 60.0  # For time-dependent metrics
```

### 2.2 Initialize and Fit (Phase 1 & 2)

```python
# Phase 1: Initialize
model = factory.create(
    "survival",
    data=df,
    duration_var=DURATION_VAR,
    dependent_var=EVENT_VAR,
    independent_vars=PREDICTORS
)

# Phase 2: Fit
model.fit(
    labels={'age': 'Age (years)', 'bmi': 'Body Mass Index'},
    penalizer=0.1
)
```

### 2.3 Summary and Diagnostics (Phase 3)

```python
# Phase 3: Generate outputs
PLOTS = [
    'forest_plot',           # Hazard ratios
    'schoenfeld_residuals',  # PH assumption
    'martingale_residuals',  # Linearity check
    'roc_auc',               # Discrimination
    'calibration_plot'       # Calibration
]

model.summary(
    assumptions=True,
    performance=True,
    plots=PLOTS,
    target_time=TARGET_TIME
)
```

## 3. Same Pattern, Different Models

| Model | `create()` call |
|-------|-----------------|
| Logistic Regression | `factory.create("logistic", data=df, dependent_var="outcome", independent_vars=[...])` |
| GLM | `factory.create("glm", data=df, dependent_var="outcome", independent_vars=[...], family="gaussian")` |
| Survival | `factory.create("survival", data=df, duration_var="time", dependent_var="event", independent_vars=[...])` |
| MICE | `MICEImputer(n=5, max_iter=10).fit(df)` |

All follow: **Initialize → Fit → Summary**

---

## 4. Interpreting Key Outputs

### Hazard Ratio Table

| HR Value | Interpretation |
|----------|----------------|
| HR > 1 | Covariate **increases** risk (shorter survival) |
| HR < 1 | Covariate **decreases** risk (longer survival) |
| HR = 1 | No effect |

### Diagnostic Plots

| Plot | What it Checks |
|------|----------------|
| `schoenfeld_residuals` | Proportional Hazards assumption |
| `martingale_residuals` | Linearity of continuous predictors |
| `roc_auc` | Discrimination at specific time point |
| `calibration_plot` | Agreement between predicted and observed |

### Key Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| C-Index | 0.5 – 1.0 | 0.5 = chance, 1.0 = perfect |
| AUC-ROC | 0.5 – 1.0 | Discrimination at `target_time` |
| Brier Score | 0 – 0.25 | Lower = better calibration |



##  Next Steps

| Want to... | Go to... |
|------------|----------|
| Deep dive into Survival theory? | **[Survival Tutorial](../user_guide/guide_survival.md)** |
| Quick reference for Survival? | **[Survival Quick Guide](quick_survival.md)** |
| See a complete example? | **[Survival Example](../examples/survival_example.ipynb)** |
| Try another model? | **[Logistic Quick Guide](quick_logistic.md)** |






