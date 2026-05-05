# ⏳ Survival Analysis (Cox PH)

!!! abstract "TL;DR"
    **Assinatura:** `factory.create("survival", data=df, duration_var="time", dependent_var="event", independent_vars=[...])`
    **O que faz:** Análise de sobrevivência com modelo Cox Proportional Hazards.
    **Quando usar:** Dados time-to-event (mortalidade, readmissão, tempo até alta).
    ⚠️ **Pré-requisito:** `dependent_var` deve ser binário (1 = evento, 0 = censura).

---

## ⚡ Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `create()` | Initialize model | `factory.create("survival", data=df, duration_var="time", dependent_var="event", independent_vars=["age", "sex"])` |
| `fit()` | Train model | `model.fit(labels={"age": "Age"}, penalizer=0.1)` |
| `summary()` | Display results | `model.summary(plots=["forest_plot", "roc_auc"])` |

---

## 🔧 Parameters - create()

| Parameter | Type | Default | Notes | Methodological Stage |
|-----------|------|---------|-------|----------------------|
| `data` | DataFrame | required | Dataset | Preprocessing |
| `duration_var` | str | required | Time-to-event column | Modeling |
| `dependent_var` | str | required | Event indicator (1=event, 0=censored) | Modeling |
| `independent_vars` | list | None | Predictors | Modeling |

---

## 🔧 Parameters - fit()

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `formula` | str | None | R-style formula: `"time + event ~ age + sex"` | Modeling |
| `labels` | dict | None | Readable names: `{"age": "Age (years)"}` | Evaluation |
| `penalizer` | float | `0.1` | L2 regularization strength | Modeling |

---

## 🔧 Parameters - summary()

| Parameter | Type | Default | Options | Methodological Stage |
|-----------|------|---------|---------|----------------------|
| `assumptions` | bool | `False` | Show VIF, outliers, PH test | Evaluation |
| `performance` | bool | `False` | Show Accuracy, Precision, Recall, F1, AUC | Evaluation |
| `plots` | list | None | `["forest_plot", "roc_auc", "martingale"]` | Evaluation |
| `target_time` | float | None | Time point for ROC AUC calculation | Evaluation |

---

## 📊 Main Attributes (post-fit)

| Attribute | Content |
|-----------|---------|
| `model.summary_df` | Hazard Ratios, CI 95%, p-values |
| `model.performance_metrics_df` | Accuracy, Precision, Recall, F1, AUC |
| `model.c_index` | Concordance index |
| `model.baseline_hazard` | Baseline hazard function |
| `model.brier_score` | Brier score at target time |

---

## 📈 Common Metrics

| Category | Metric | Interpretation |
|----------|--------|----------------|
| Effect Size | Hazard Ratio (HR) | HR > 1 = increased risk |
| Discrimination | C-index | 0.5 = chance, 1.0 = perfect |
| Discrimination | AUC-ROC | Time-dependent AUC |
| Calibration | Brier Score | Lower = better |
| Assumption | VIF | > 5 = multicollinearity |

---

## 📈 Hazard Ratio Interpretation

| HR Value | Interpretation |
|----------|----------------|
| HR > 1 | Predictor **increases** hazard (higher risk, shorter survival) |
| HR < 1 | Predictor **decreases** hazard (lower risk, longer survival) |
| HR = 1 | No effect on survival |

---

## 🎯 Minimal Example

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

# Create model
model = factory.create(
    "survival",
    data=df,
    duration_var="time_to_event",
    dependent_var="event_death",
    independent_vars=["age", "sex", "bmi"]
)

# Train
model.fit(
    labels={"age": "Age", "bmi": "BMI"},
    penalizer=0.05
)

# View results
model.summary(
    assumptions=True,
    performance=True,
    plots=["forest_plot", "roc_auc"],
    target_time=28
)

# Access hazard ratios
print(model.summary_df)
```

## 🎯 Formula Example

```python
# Using formula notation with interaction
model = factory.create(
    "survival",
    data=df,
    duration_var="time",
    dependent_var="event",
    independent_vars=["age", "sex", "bmi"]
)

model.fit(formula="time + event ~ age * sex + bmi")
```

## 🔗 Quick Links

| Want to... | Go to... |
|------------|----------|
| Understand the theory? | **[Survival Tutorial](../user_guide/guide_survival.md)** |