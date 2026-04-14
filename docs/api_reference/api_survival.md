# Survival Analysis

!!! abstract "Quick Reference"
    | Method | Description | Example |
    |--------|-------------|---------|
    | `create()` | Initialize survival model | `factory.create("survival", ...)` |
    | `fit()` | Train model | `model.fit()` |
    | `summary()` | Display results | `model.summary()` |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | pd.DataFrame | required | Dataset |
| `time_col` | str | required | Time to event |
| `event_col` | str | required | Event indicator (0/1) |