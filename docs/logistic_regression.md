# Logistic Regression

The `RAPID_LogisticRegression` pipeline provides a full logistic regression analysis for binary outcomes. It is built on a Generalised Linear Model (GLM) framework using the Binomial family, and supports multiple link functions to accommodate different modelling assumptions. It is suited for clinical and epidemiological research where the outcome of interest is a binary event, such as mortality, readmission, or disease presence.

!!! note "Outcome variable requirements"
    The outcome variable must be binary and coded strictly as `0` and `1`. Any other coding will raise a validation error before the model is fitted.

---

## Initialisation

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

model = factory.create(
    "logistic",
    data=df,
    yvar="outcome",
    predictors=["age", "sex", "bmi"],
    regression_type="Multi"
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | required | The dataset to analyse. Must contain all outcome and predictor columns. Rows with missing values are dropped automatically. |
| `yvar` | `str` | `None` | The name of the outcome (dependent) variable column. Must be binary and coded as 0/1. Required if `formula` is not provided. |
| `predictors` | `list` | `None` | A list of predictor (independent) variable column names. Required if `formula` is not provided. |
| `formula` | `str` | `None` | A Patsy-style formula string (e.g. `"outcome ~ age + sex"`). If provided, `yvar` and `predictors` are not required. |
| `family` | `str` | `"binomial"` | The distributional family for the GLM. See [Supported Families and Links](#supported-families-and-links). |
| `link` | `str` | `"logit"` | The link function for the GLM. See [Supported Families and Links](#supported-families-and-links). |
| `regression_type` | `str` | `"Multi"` | Either `"Multi"` for multivariable regression or `"Uni"` for univariable regression. Affects column naming in the results table. |
| `classification_threshold` | `float` | `0.5` | Probability threshold used to convert predicted probabilities into binary class predictions for classification metrics. |

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
    plots=["forest_plot", "roc_curve", "confusion_matrix"],
    vif_threshold=5.0
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `assumptions` | `str` or `list` | `None` | Pass `"all"` to show all assumption tests, or a list of specific test names. `None` skips this section. |
| `performance` | `str` or `list` | `None` | Pass `"all"` to show all performance metrics, or a list of specific metric names. `None` skips this section. |
| `cross_val` | `str` or `list` | `None` | Pass `"all"` to show all cross-validation metrics, or a list of specific metric names. `None` skips this section. |
| `plots` | `list` | `None` | A list of plot names to display. Options: `"forest_plot"`, `"roc_curve"`, `"confusion_matrix"`. |
| `vif_threshold` | `float` | `5.0` | The threshold above which a VIF value is flagged as indicating multicollinearity. |

---

## Assumption Tests

Assumption tests are run automatically during `fit()`. Results are accessible via `model.assumption_metrics_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| Events Per Variable (EPV) | The number of outcome events per predictor variable. Values below 10 may lead to unstable coefficient estimates. | `model.epv` |
| Influential Outliers Threshold | Cook's distance threshold (4/n). Observations exceeding this are flagged as influential. | `model.influential_outliers_threshold` |
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
model.summary(assumptions=["Events Per Variable (EPV)", "VIF", "Influential Outliers"])
```

Available names: `"Events Per Variable (EPV)"`, `"Influential Outliers Threshold"`, `"Number of Influential Points"`, `"VIF"`, `"Influential Outliers"`.

---

## Performance Metrics

Performance metrics are computed automatically during `fit()`. Results are accessible via `model.performance_metrics_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| Accuracy | Proportion of correctly classified observations, using `classification_threshold`. | `model.accuracy` |
| Log Loss | Measures the uncertainty of predicted probabilities. Lower is better. | `model.logloss` |
| Precision | Proportion of predicted positives that are true positives. | `model.precision` |
| Recall | Proportion of actual positives correctly identified. | `model.recall` |
| F1 Score | Harmonic mean of precision and recall. | `model.f1` |
| AUC-ROC | Area under the receiver operating characteristic curve. Measures discrimination ability independent of threshold. | `model.auc` |
| AIC | Akaike Information Criterion | `model.aic` |
| BIC | Bayesian Information Criterion | `model.bic` |
| LLF | Log-likelihood of the fitted model | `model.llf` |
| McFadden R² | Pseudo R² based on log-likelihood ratio | `model.mcfadden_r2` |
| Adjusted McFadden R² | McFadden R² penalised for model complexity | `model.mcfadden_adj_r2` |
| Efron R² | Pseudo R² based on residual sum of squares | `model.efron_r2` |
| Cox-Snell R² | Pseudo R² based on likelihood ratio, does not reach 1 for perfect models | `model.cox_snell_r2` |
| Nagelkerke R² | Scaled version of Cox-Snell R², can reach 1 | `model.nagelkerke_r2` |
| Tjur R² | Mean predicted probability for events minus mean predicted probability for non-events | `model.tjur_r2` |
| Confusion Matrix | Counts of true negatives, false positives, false negatives, and true positives | `model.cm` |

### Performance Metrics Dataframe

```python
model.performance_metrics_df
```

### Confusion Matrix

The confusion matrix is stored as a 2x2 NumPy array and can be accessed directly:

```python
model.cm  # [[TN, FP], [FN, TP]]
```

### Selecting Specific Performance Metrics

```python
model.summary(performance=["Accuracy", "AUC-ROC", "F1 Score", "AIC", "Confusion Matrix"])
```

---

## Cross-Validation Metrics

Cross-validation is run automatically during `fit()` when `cross_val=True`. Results are accessible via `model.cv_df`.

### Available Metrics

| Metric | Description | Accessed via |
|--------|-------------|--------------|
| Mean Accuracy | Average classification accuracy across all folds | `model.cross_val_scores.mean()` |
| Standard Deviation | Variability of accuracy across folds | `model.cross_val_scores.std()` |
| Individual Fold Accuracies | Accuracy for each individual fold | `model.cross_val_scores` |

### Cross-Validation Dataframe

```python
model.cv_df
```

### Selecting Specific CV Metrics

```python
model.summary(cross_val=["Mean Accuracy", "Standard Deviation"])
```

---

## Results Summary

The odds ratio table produced after fitting is stored in `model.summary_df`. It contains one row per predictor and the following columns:

| Column | Description |
|--------|-------------|
| `Variable` | Predictor name (or human-readable label if `labels` were provided) |
| `OddsRatio (multi)` / `OddsRatio (uni)` | Exponentiated coefficient, interpreted as an odds ratio |
| `LowerCI (multi)` / `LowerCI (uni)` | Lower bound of 95% confidence interval for the odds ratio |
| `UpperCI (multi)` / `UpperCI (uni)` | Upper bound of 95% confidence interval for the odds ratio |
| `p-value (multi)` / `p-value (uni)` | p-value for the coefficient |

The `(multi)` or `(uni)` suffix depends on the `regression_type` set at initialisation.

```python
model.summary_df
```

---

## Supported Families and Links

Logistic regression uses the Binomial family exclusively. The following link function combinations are supported and tested:

| Family | Link | Description |
|--------|------|-------------|
| `binomial` | `logit` | Standard logistic regression. Models the log-odds of the outcome. Default and most widely used. |
| `binomial` | `probit` | Models the outcome via the inverse normal CDF. Common in econometrics and some epidemiological contexts. |
| `binomial` | `cloglog` | Complementary log-log link. Appropriate when the probability of the event is very low or when the outcome has an underlying extreme value distribution. |

---

## Plots

The following plots can be requested via `summary(plots=[...])`:

| Plot name | Description |
|-----------|-------------|
| `"forest_plot"` | Displays odds ratios and 95% confidence intervals for each predictor on a log scale, with a reference line at 1 |
| `"roc_curve"` | Receiver operating characteristic curve showing the trade-off between sensitivity and specificity across all thresholds, with AUC displayed |
| `"confusion_matrix"` | Heatmap of true negatives, false positives, false negatives, and true positives at the specified `classification_threshold` |