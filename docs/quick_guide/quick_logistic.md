# 📊 Logistic Regression

!!! abstract "TL;DR"
    **Assinatura:** `factory.create("logistic", data=df, dependent_var="outcome", independent_vars=[...])`
    **O que faz:** Regressão logística para desfechos binários.
    **Quando usar:** Predizer eventos como mortalidade, readmissão ou presença de doença.
    ⚠️ **Pré-requisito:** Variável dependente deve ser estritamente `0` e `1`.

---

## ⚡ Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `create()` | Initialize model | `factory.create("logistic", data=df, dependent_var="death", independent_vars=["age", "sex"])` |
| `fit()` | Train model | `model.fit(labels={"age": "Age"}, cross_val=True)` |
| `summary()` | Display results | `model.summary(plots=["forest_plot", "roc_curve"])` |

---

## 🔧 Parameters - create()

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `data` | DataFrame | required | Dataset |
| `dependent_var` | str | None | **Must be 0/1** |
| `independent_vars` | list | None | Predictors |
| `formula` | str | None | Alternative: `"y ~ x1 + x2"` |
| `link` | str | `"logit"` | `"logit"`, `"probit"`, `"cloglog"` |
| `classification_threshold` | float | `0.5` | Threshold for binary predictions |

---

## 🔧 Parameters - fit()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | dict | None | Readable names: `{"age": "Age (years)"}` |
| `cross_val` | bool | True | Enable cross-validation |
| `n_splits` | int | 5 | Number of folds |

---

## 🔧 Parameters - summary()

| Parameter | Type | Default | Common Values |
|-----------|------|---------|---------------|
| `assumptions` | str/list | None | `"all"`, `["VIF", "EPV"]` |
| `performance` | str/list | None | `"all"`, `["AUC", "AIC", "F1"]` |
| `plots` | list | None | `["forest_plot", "roc_curve", "confusion_matrix"]` |
| `vif_threshold` | float | 5.0 | Multicollinearity alert |

---

## 📊 Main Attributes (post-fit)

| Attribute | Content |
|-----------|---------|
| `model.summary_df` | Odds Ratios, CI 95%, p-values |
| `model.performance_metrics_df` | AUC, AIC, BIC, Accuracy, F1, R² |
| `model.vif_df` | Variance Inflation Factor per predictor |
| `model.cv_df` | Cross-validation metrics |
| `model.cm` | Confusion matrix (2×2 array) |

---

## 📈 Common Metrics

| Category | Metric | Attribute |
|----------|--------|-----------|
| Discrimination | AUC-ROC | `model.auc` |
| Fit | AIC / BIC | `model.aic` / `model.bic` |
| Classification | F1 Score | `model.f1` |
| Classification | Accuracy | `model.accuracy` |
| Pseudo R² | McFadden | `model.mcfadden_r2` |
| Pseudo R² | Tjur | `model.tjur_r2` |
| Assumption | EPV | `model.epv` |
| Multicollinearity | VIF | `model.vif_df` |

---

## 📈 Supported Links

| Link | Description | Interpretation |
|------|-------------|----------------|
| `logit` | Standard logistic regression (default) | Odds Ratios |
| `probit` | Inverse normal CDF | Not odds ratios |
| `cloglog` | Complementary log-log | Approximates hazard ratios |

---

## 🎯 Minimal Example

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

# Create model
model = factory.create(
    "logistic",
    data=df,
    dependent_var="mortality",
    independent_vars=["age", "bmi", "sex"]
)

# Train
model.fit(labels={"age": "Age", "bmi": "BMI"})

# View results
model.summary(
    assumptions=["VIF", "EPV"],
    performance=["AUC", "AIC", "Accuracy"],
    plots=["forest_plot", "roc_curve"]
)

# Access odds ratios
print(model.summary_df)
```

## 🔗 Quick Links

| Want to... | Go to... |
|------------|----------|
| Understand the theory? | **[Logistic Regression Tutorial](../user_guide/guide_logistic.md)** |