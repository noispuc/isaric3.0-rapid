# 📊 GLM (Generalized Linear Model)

!!! abstract "TL;DR"
    **Assinatura:** `factory.create("glm", data=df, dependent_var="outcome", independent_vars=[...])`
    **O que faz:** Regressão linear generalizada para desfechos contínuos.
    **Quando usar:** Modelar relações lineares com diferentes distribuições (Gaussian, Gamma, Tweedie).

---

## ⚡ Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `create()` | Initialize GLM | `factory.create("glm", data=df, dependent_var="y", independent_vars=["x1", "x2"])` |
| `fit()` | Train model | `model.fit(labels={"x1": "Predictor 1"}, cross_val=True)` |
| `summary()` | Display results | `model.summary(plots=["forest_plot", "qq_plot"])` |

---

## 🔧 Parameters - create()

| Parameter | Type | Default | Notes | Methodological Stage |
|-----------|------|---------|-------|----------------------|
| `data` | DataFrame | required | Dataset | Preprocessing |
| `dependent_var` | str | None | Outcome variable | Modeling |
| `independent_vars` | list | None | Predictors | Modeling |
| `formula` | str | None | Alternative: `"y ~ x1 + x2"` | Modeling |
| `family` | str | `"gaussian"` | `"gaussian"`, `"gamma"`, `"inv_gaussian"`, `"tweedie"` | Modeling |
| `link` | str | `"identity"` | `"identity"`, `"log"`, `"inverse"`, `"sqrt"` | Modeling |

---

## 🔧 Parameters - fit()

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `labels` | dict | None | Readable names: `{"age": "Age (years)"}` | Evaluation |
| `cross_val` | bool | True | Enable cross-validation | Validation |
| `n_splits` | int | 5 | Number of folds | Validation |

---

## 🔧 Parameters - summary()

| Parameter | Type | Default | Common Values | Methodological Stage |
|-----------|------|---------|---------------|----------------------|
| `assumptions` | str/list | None | `"all"`, `["Durbin-Watson", "VIF"]` | Evaluation |
| `performance` | str/list | None | `"all"`, `["R2", "AIC", "RMSE"]` | Evaluation |
| `plots` | list | None | `["forest_plot", "residuals_vs_fitted", "qq_plot"]` | Evaluation |
| `vif_threshold` | float | 5.0 | Multicollinearity alert threshold | Evaluation |

---

## 📊 Main Attributes (post-fit)

| Attribute | Content |
|-----------|---------|
| `model.summary_df` | Coefficients, CI 95%, p-values |
| `model.performance_metrics_df` | R², AIC, BIC, RMSE, MSE, MAE |
| `model.vif_df` | Variance Inflation Factor per predictor |
| `model.cv_df` | Cross-validation metrics |
| `model.assumption_metrics_df` | Durbin-Watson, Shapiro-Wilk, influential points |

---

## 📈 Common Metrics

| Category | Metric | Attribute |
|----------|--------|-----------|
| Fit | R² / Adjusted R² | `model.r2` / `model.adjusted_r2` |
| Fit | AIC / BIC | `model.aic` / `model.bic` |
| Error | RMSE | `model.rmse` |
| Error | MAE | `model.mae` |
| Assumption | Durbin-Watson | `model.dw` |
| Assumption | Shapiro-Wilk p-value | `model.shapiro_wilk_p_value` |
| Multicollinearity | VIF | `model.vif_df` |

---

## 📈 Supported Families & Links

| Family | Link | Use Case |
|--------|------|----------|
| `gaussian` | `identity` | Standard linear regression |
| `gamma` | `log` | Positively skewed, positive outcomes |
| `tweedie` | `log` | Mixed zeros + positive continuous |

---

## 🎯 Minimal Example

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

# Create model
model = factory.create(
    "glm",
    data=df,
    dependent_var="outcome",
    independent_vars=["age", "bmi", "sex"],
    family="gaussian",
    link="identity"
)

# Train
model.fit(labels={"age": "Age", "bmi": "BMI"})

# View results
model.summary(
    assumptions=["VIF", "Durbin-Watson"],
    performance=["R2", "AIC", "RMSE"],
    plots=["forest_plot"]
)

# Access coefficients
print(model.summary_df)
```

## 🔗 Quick Links

| Want to... | Go to... |
|------------|----------|
| Understand the theory? | **[GLM Tutorial](../user_guide/guide_glm.md)** |