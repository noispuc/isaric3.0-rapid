# Linear Regression

The `RAPID_GLM` pipeline provides a full linear regression analysis for continuous outcomes. It is built on a Generalised Linear Model (GLM) framework, meaning it supports several distributional families beyond the standard Gaussian, making it suitable for a range of continuous outcome types encountered in clinical and epidemiological research.

---

## Initialisation

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

model = factory.create(
    "glm",
    data=df,
    dependent_var="outcome",
    independent_vars=["age", "sex", "bmi"],
    regression_type="Multi"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | The dataset to analyse. Must contain all outcome and predictor columns. Rows with missing values are dropped automatically. |
| `dependent_var` | `str` | `None` | The name of the outcome (dependent) variable column. Required if `formula` is not provided. |
| `independent_vars` | `list` | `None` | A list of predictor (independent) variable column names. Required if `formula` is not provided. |
| `formula` | `str` | `None` | A Patsy-style formula string (e.g. `"outcome ~ age + sex"`). If provided, `dependent_var` and `independent_vars` are not required. |
| `family` | `str` | `"gaussian"` | The distributional family for the GLM. See [Supported Families and Links](#supported-families-and-links). |
| `link` | `str` | `"identity"` | The link function for the GLM. See [Supported Families and Links](#supported-families-and-links). |
| `regression_type` | `str` | `"Multi"` | Either `"Multi"` for multivariable regression or `"Uni"` for univariable regression. Affects column naming in the results table. |

---

## fit()

Fits the model and runs all evaluation steps, including assumption testing, performance metrics, and optionally cross-validation.

```python
model.fit(
    labels={"age": "Age (years)", "sex": "Sex"},
    cross_val=True,
    n_splits=5
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `labels` | `dict` | `None` | A dictionary mapping raw variable names to human-readable labels for display in result tables. |
| `cross_val` | `bool` | `True` | Whether to perform k-fold cross-validation after fitting. |
| `n_splits` | `int` | `5` | Number of folds for cross-validation. Only used if `cross_val=True`. |

---

## summary()

Displays results after fitting. All arguments are optional — pass only what you want to see.

```python
model.summary(
    assumptions="all",
    performance="all",
    cross_val="all",
    plots=["forest_plot", "residuals_vs_fitted", "qq_plot"],
    vif_threshold=5.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assumptions` | `str` or `list` | `None` | Pass `"all"` to show all assumption tests, or a list of specific test names. `None` skips this section. |
| `performance` | `str` or `list` | `None` | Pass `"all"` to show all performance metrics, or a list of specific metric names. `None` skips this section. |
| `cross_val` | `str` or `list` | `None` | Pass `"all"` to show all cross-validation metrics, or a list of specific metric names. `None` skips this section. |
| `plots` | `list` | `None` | A list of plot names to display. Options: `"forest_plot"`, `"residuals_vs_fitted"`, `"qq_plot"`. |
| `vif_threshold` | `float` | `5.0` | The threshold above which a VIF value is flagged as indicating multicollinearity. |

---

## Assumption Tests

Assumption tests are run automatically during `fit()`. Results are accessible via `model.assumption_metrics_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| Durbin-Watson | Tests for autocorrelation in residuals. Values close to 2 suggest independence. Values below 1.5 suggest positive autocorrelation; above 2.5 suggest negative autocorrelation. | `model.dw` |
| Shapiro-Wilk Statistic | Test statistic for normality of residuals. | `model.shapiro_wilk_test_statistic` |
| Shapiro-Wilk p-value | p-value for the Shapiro-Wilk test. A value below 0.05 suggests residuals are not normally distributed. | `model.shapiro_wilk_p_value` |
| Influential Outliers Threshold | Cook's distance threshold (4/n). Points exceeding this are flagged as influential. | `model.influential_outliers_threshold` |
| Number of Influential Points | Count of observations with Cook's distance above the threshold. | `model.influential_points` |

### Assumption Metrics Dataframe

All assumption metrics are stored together in a single dataframe:

```python
model.assumption_metrics_df
```

### VIF Dataframe

Variance Inflation Factors for each predictor are stored separately:

```python
model.vif_df
```

VIF values above `vif_threshold` (default 5.0) are flagged as indicating potential multicollinearity.

### Selecting Specific Assumption Tests

Pass a list of test names to `summary()` to display only those tests:

```python
model.summary(assumptions=["Durbin-Watson", "Shapiro-Wilk p-value", "VIF", "Influential Outliers"])
```

Available names: `"Durbin-Watson"`, `"Shapiro-Wilk Statistic"`, `"Shapiro-Wilk p-value"`, `"Influential Outliers Threshold"`, `"Number of Influential Points"`, `"VIF"`, `"Influential Outliers"`.

---

## Performance Metrics

Performance metrics are computed automatically during `fit()`. Results are accessible via `model.performance_metrics_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| MSE | Mean Squared Error | `model.mse` |
| RMSE | Root Mean Squared Error | `model.rmse` |
| MAE | Mean Absolute Error | `model.mae` |
| R² | Coefficient of determination | `model.r2` |
| Adjusted R² | R² adjusted for number of independent_vars | `model.adjusted_r2` |
| McFadden R² | Pseudo R² based on log-likelihood ratio | `model.mcfadden_r2` |
| Adjusted McFadden R² | McFadden R² penalised for model complexity | `model.mcfadden_adj_r2` |
| Efron R² | Pseudo R² based on residual sum of squares | `model.efron_r2` |
| AIC | Akaike Information Criterion | `model.aic` |
| BIC | Bayesian Information Criterion | `model.bic` |
| LLF | Log-likelihood of the fitted model | `model.llf` |

### Performance Metrics Dataframe

```python
model.performance_metrics_df
```

### Selecting Specific Performance Metrics

```python
model.summary(performance=["MSE", "RMSE", "R2", "AIC"])
```

---

## Cross-Validation Metrics

Cross-validation is run automatically during `fit()` when `cross_val=True`. Results are accessible via `model.cv_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| Mean CV MSE | Average MSE across all folds | `model.cv_mse_scores.mean()` |
| Standard Deviation of CV MSE | Variability of MSE across folds | `model.cv_mse_scores.std()` |
| Individual Fold MSEs | MSE for each individual fold | `model.cv_mse_scores` |

### Cross-Validation Dataframe

```python
model.cv_df
```

### Selecting Specific CV Metrics

```python
model.summary(cross_val=["Mean CV MSE", "Standard Deviation of CV MSE"])
```

---

## Results Summary

The coefficient table produced after fitting is stored in `model.summary_df`. It contains one row per predictor and the following columns:

| Column | Description |
|--------|-------------|
| `Variable` | Predictor name (or human-readable label if `labels` were provided) |
| `Coefficient (multi)` / `Coefficient (uni)` | Regression coefficient estimate |
| `LowerCI (multi)` / `LowerCI (uni)` | Lower bound of 95% confidence interval |
| `UpperCI (multi)` / `UpperCI (uni)` | Upper bound of 95% confidence interval |
| `p-value (multi)` / `p-value (uni)` | p-value for the coefficient |

The `(multi)` or `(uni)` suffix depends on the `regression_type` set at initialisation.

```python
model.summary_df
```

---

## Supported Families and Links

The `family` and `link` parameters control the distributional assumptions of the GLM. The following combinations are supported and tested:

| Family | Link | Use case |
|--------|------|----------|
| `gaussian` | `identity` | Standard linear regression for normally distributed outcomes (default) |
| `gamma` | `log` | Positively skewed, strictly positive continuous outcomes |
| `gamma` | `inverse` | Alternative for positive continuous outcomes |
| `inv_gaussian` | `inverse` | Heavily right-skewed positive outcomes |
| `tweedie` | `log` | Outcomes with a mix of zeros and positive continuous values |

### All Available Links

Not all links are appropriate for all families. The full set of available link functions is: `"identity"`, `"log"`, `"inverse"`, `"sqrt"`.

---

## Plots

The following plots can be requested via `summary(plots=[...])`:

| Plot name | Description |
|-----------|-------------|
| `"forest_plot"` | Displays coefficient estimates and 95% confidence intervals for each predictor |
| `"residuals_vs_fitted"` | Scatter plot of residuals against fitted values, used to assess homoscedasticity and linearity |
| `"qq_plot"` | Quantile-quantile plot of residuals against a normal distribution, used to assess normality of errors |