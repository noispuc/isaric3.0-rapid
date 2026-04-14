# LogisticRegression

!!! abstract "Quick Reference"
    | Method | Description | Example |
    |--------|-------------|---------|
    | `create()` | Initialize model | `factory.create("logistic", ...)` |
    | `fit()` | Train model | `model.fit()` |
    | `summary()` | Display results | `model.summary()` |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | required | Dataset |
| `dependent_var` | str | None | Outcome (0/1) |
| `independent_vars` | list | None | Predictors |

## Methods

### `fit()`

[Documentação gerada automaticamente ou manual]

### `summary()`

[Documentação gerada automaticamente ou manual]

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `summary_df` | pd.DataFrame | Odds ratios table |
| `performance_metrics_df` | pd.DataFrame | AUC, AIC, etc. |