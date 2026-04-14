# GLM (Generalized Linear Model)

!!! abstract "Quick Reference"
    | Method | Description | Example |
    |--------|-------------|---------|
    | `create()` | Initialize GLM | `factory.create("glm", ...)` |
    | `fit()` | Train model | `model.fit()` |
    | `summary()` | Display results | `model.summary()` |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | required | Dataset |
| `dependent_var` | str | None | Outcome variable |
| `family` | str | "gaussian" | Distribution family |
| `link` | str | None | Link function |

## Supported Families and Links

| Family | Link | Use Case |
|--------|------|----------|
| `gaussian` | `identity` | Linear regression |
| `binomial` | `logit` | Logistic regression |
| `poisson` | `log` | Count data |