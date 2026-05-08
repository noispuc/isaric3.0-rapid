#  Quickstart Guide

The **RAPID Pipeline** is designed as a **modular and extensible structure** for implementing various statistical and machine learning models for clinical research.

This guide will help you run your first analysis in minutes.

---

## 1. Core Principles and Modularity

The pipeline is built on a simple **three-phase architecture**, which is universal for all models:

| Phase | Method | Role in the Pipeline |
|:------|:-------|:---------------------|
| **Phase 1: Initialization** | `create()` | Selects the model type (e.g., Logistic, GLM, Survival) and configures parameters. |
| **Phase 2: Training** | `.fit()` | Performs the training procedure, including preprocessing and model fitting. |
| **Phase 3: Diagnostics** | `.summary()` | Generates performance metrics, assumption tests, and diagnostic plots. |

This architecture ensures **consistency** across all models: `create()` → `fit()` → `summary()`.

---

## 2. Prerequisites

Before you begin, make sure you have:

- **[Installed RAPID](installation.md)** 
- Python 3.10 or higher
- A dataset ready for analysis (CSV, Excel, or DataFrame)

---

## 3. Your First Analysis (5 Minutes)

### 3.1 Import and Initialize

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

# Create a factory instance
factory = RAPID_PipelineFactory()
```

### 3.2 Load Your Data

```python
import pandas as pd

# Load your dataset
df = pd.read_csv("your_data.csv")

# Example: Predict mortality using age, sex, and BMI
model = factory.create(
    "logistic",
    data=df,
    dependent_var="mortality",
    independent_vars=["age", "sex", "bmi"]
)
```

### 3.3 Train the Model

```python
# Fit the model with optional labels for readability
model.fit(
    labels={
        "age": "Age (years)",
        "sex": "Sex",
        "bmi": "Body Mass Index"
    },
    cross_val=True,
    n_splits=5
)
```

### 3.4 View Results

```python
# Display a comprehensive summary
model.summary(
    assumptions=["VIF", "EPV"],
    performance=["AUC", "AIC", "Accuracy"],
    plots=["forest_plot", "roc_curve"]
)
```

### 3.5 Access Results Programmatically

```python
# Odds ratios and confidence intervals
odds_ratios = model.summary_df
print(odds_ratios)

# Performance metrics
metrics = model.performance_metrics_df
print(metrics)

# Confusion matrix
cm = model.cm
print(cm)
```
---

## 4. Supported Models

| Model | `model_type` | Use Case |
|-------|--------------|----------|
| Logistic Regression | `"logistic"` | Binary outcomes (0/1) |
| GLM | `"glm"` | Generalized Linear Models |
| Survival Analysis | `"survival"` | Time-to-event data |
| MICE Imputation | `"mice"` | Missing data handling |

---

##  Need Help?

- **[ GitHub Issues](https://github.com/noispuc/isaric3.0-rapid/issues)** – Report bugs or request features.
- **[ Contact Support](mailto:data@isaric.org)** – Get help from the ISARIC team.
