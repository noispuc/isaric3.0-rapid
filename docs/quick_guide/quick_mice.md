# MICE Imputation

!!! abstract "Quick Reference"
    | Method | Description | Example |
    |--------|-------------|---------|
    | `create()` | Initialize MICE | `factory.create("mice", ...)` |
    | `fit()` | Run imputation | `model.fit()` |
    | `transform()` | Apply imputation | `model.transform()` |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | required | Dataset with missing values |
| `m` | int | 5 | Number of imputations |
| `max_iter` | int | 10 | Iterations per imputation |