# Generalized Linear Model (GLM)

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

## Statistical Notes

**1. The Generalised Linear Model**

The GLM models the relationship between a set of predictors and an outcome whose distribution belongs to the exponential family. It is defined by three components:

*   A **random component**: the conditional distribution of the outcome Y, assumed to belong to the exponential family (Gaussian, Gamma, Inverse Gaussian, Tweedie).
*   A **systematic component**: a linear predictor $\eta = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$.
*   A **link function** $g(\cdot)$ such that $g(\mu) = \eta$, connecting the mean of the distribution to the linear predictor.

The Gaussian family with an identity link recovers standard OLS. Parameters are estimated by maximum likelihood via iteratively reweighted least squares (IRLS).

---

**Interpreting the GLM**

**1. Coefficients**

Under an **identity link** (Gaussian), $\beta_j$ is the expected change in $Y$ per unit increase in $X_j$, holding all other predictors constant.

Under a **log link** (Gamma, Tweedie), the coefficient operates on the log-mean scale. Exponentiating recovers a ratio of means:

$$
\text{Mean ratio} = e^{\beta_j}
$$

*   If the mean ratio $> 1$, the predictor increases the outcome mean.
*   If the mean ratio $< 1$, the predictor decreases the outcome mean.
*   If the mean ratio $= 1$, the predictor has no effect.

The pipeline labels the exponentiated coefficient according to the family and link combination: **Mean Ratio** for Gamma and Tweedie with a log link, and **Coefficient** for identity link models. These labels are reflected in the forest plot axis and the results table. Coefficients from models with different effect labels are not directly comparable.

**2. Confidence Intervals**

The 95% confidence interval for a coefficient is:

$$
\text{CI} = \left[ \beta_j - 1.96 \cdot \text{SE}(\beta_j),\ \beta_j + 1.96 \cdot \text{SE}(\beta_j) \right]
$$

where $\text{SE}(\beta_j)$ is derived from the Fisher information matrix. Under a log link, exponentiate both bounds to obtain the CI on the ratio scale. If the CI excludes zero (identity link) or one (log link), the predictor is statistically significant at $\alpha = 0.05$.

**3. p-value**

The p-value tests the null hypothesis that $\beta_j = 0$:

*   If $p < 0.05$, the predictor has a statistically significant association with the outcome.
*   If $p > 0.05$, there is insufficient evidence to conclude an association exists.

Statistical significance does not imply clinical relevance. Effect size and confidence interval width should always be considered alongside the p-value.

---

**Performance Metrics**

*   **R² and Adjusted R²**: Proportion of variance in $Y$ explained by the model. Only has its conventional interpretation under Gaussian/identity. Adjusted R² penalises for the number of predictors and should be preferred when comparing models of different sizes.

*   **Pseudo R²**: For non-Gaussian families, McFadden R² ($1 - \text{LLF} / \text{LL}_0$) and Efron R² are reported as approximate analogues. These are not directly comparable to standard R² or to each other across families. Values of McFadden R² between 0.2 and 0.4 are generally considered good fit.

*   **AIC and BIC**: AIC ($-2 \cdot \text{LLF} + 2k$) and BIC ($-2 \cdot \text{LLF} + k \cdot \log n$) penalise log-likelihood for model complexity. Lower values indicate better fit relative to the number of parameters. BIC applies a stronger penalty and is preferred when the goal is identifying the true model rather than optimising prediction.

*   **Cross-validation MSE**: The mean and standard deviation of MSE across k folds assesses in-sample stability. A low standard deviation indicates that results are not sensitive to which observations are included in fitting. This is not equivalent to external validation on an independent dataset. The fold split uses a fixed random seed (`random_state=42`), meaning results are reproducible but not sensitive to the choice of seed; analysts requiring seed sensitivity analysis should implement CV externally.

---

**Advantages**

*   Unified framework: the same pipeline handles normally distributed, skewed positive, and compound zero-positive outcomes by changing the family and link.
*   Interpretable coefficients: additive effects under identity links; multiplicative mean ratios under log links.
*   Integrated diagnostics: assumption tests, influence measures, and cross-validation are computed automatically during `fit()`.

---

**Limitations**

*   Family and link selection is the analyst's responsibility. The pipeline does not perform automatic family selection or goodness-of-fit tests to guide this choice.
*   Rows with missing values are dropped automatically (listwise deletion). This is valid only under Missing Completely At Random (MCAR). For datasets with substantial missingness, multiple imputation should be applied upstream.
*   The pipeline does not support mixed-effects GLMs, penalised regression (ridge, lasso), or time-varying covariates.

---

**Assumptions of the GLM**

The GLM relies on several critical assumptions that must hold for estimates to be valid. The pipeline provides diagnostic tools to evaluate each.

<br>

* **Correct Specification of the Linear Predictor**: The relationship between each continuous predictor and the link-transformed outcome is linear, with no relevant predictors omitted.
  * **Implications**: Misspecification causes biased, inconsistent estimates that do not diminish with larger samples.
  * **If Violated**: Add polynomial terms, interaction terms, or splines for non-linear predictors. Re-examine variable selection if omitted variable bias is suspected.
  * **Evaluation**: Inspect the residuals-vs-fitted plot (`summary(plots=["residuals_vs_fitted"])`). A systematic curve or fan pattern indicates misspecification.

<br>

* **Correct Distributional Family**: The conditional distribution of $Y | X$ belongs to the specified exponential family with the specified variance function.
  * **Implications**: Under the Gaussian family, misspecification inflates standard errors and invalidates inference. Under non-Gaussian families, mild misspecification may leave point estimates consistent but renders standard errors unreliable.
  * **If Violated**: Consider a quasi-GLM with robust standard errors, or re-specify the family.
  * **Evaluation**: Use the QQ plot (`summary(plots=["qq_plot"])`) to assess normality of residuals under the Gaussian family. For other families, examine Pearson residuals across the fitted value range.

<br>

* **Independence of Observations**: Observations must be independent of one another. This assumption may be violated in clustered or repeated-measures designs.
  * **Implications**: Violation leads to underestimated standard errors and anti-conservative p-values. Point estimates remain unbiased but inference is invalid.
  * **If Violated**: Use mixed-effects models to account for random effects, or apply cluster-robust standard errors.
  * **Evaluation**: The Durbin-Watson statistic (`model.dw`) tests for first-order autocorrelation in residuals. Values near 2 indicate independence; values below 1.5 suggest positive autocorrelation. For non-temporal data, review study design for clustering.

<br>

* **No Influential Outliers**: No individual observation exerts disproportionate leverage on the coefficient estimates.
  * **Implications**: Highly influential points can substantially shift estimated coefficients, producing results that do not generalise.
  * **If Violated**: Investigate flagged observations before removing them. Report analyses with and without the influential point.
  * **Evaluation**: Cook's distance threshold of $4/n$ is applied (`model.influential_outliers_threshold`). This threshold becomes permissive for large $n$; graphical inspection is advisable for large datasets.

<br>

* **No Multicollinearity**: Predictor variables are not strongly linearly related to one another.
  * **Implications**: High collinearity inflates standard errors and makes individual coefficient estimates unstable. Overall model fit is unaffected.
  * **If Violated**: Remove one of the correlated predictors, combine them into a composite, or use ridge regression (L2 regularisation).
  * **Evaluation**: Variance Inflation Factors are computed for each predictor (`model.vif_df`). The default flag threshold is VIF > 5.0, a widely used heuristic; some fields use 10. The threshold is configurable via `vif_threshold`.

<br>

* **Missing Completely At Random**: Rows with missing values are dropped automatically. This is valid only if the probability of missingness is unrelated to any observed or unobserved variable.
  * **Implications**: If data are Missing At Random (MAR) or Missing Not At Random (MNAR), listwise deletion produces biased estimates whose severity depends on the proportion missing and the mechanism.
  * **If Violated**: Apply multiple imputation (e.g. MICE) before passing data to the pipeline.
  * **Evaluation**: Compare the distribution of key covariates in complete vs. incomplete cases. Systematic differences suggest non-MCAR missingness.

<br>

---

### References

If you wish to further explore the Generalised Linear Model — its theoretical foundations, distributional families, estimation, and extensions — the following references are recommended:

* **Nelder, J. A., & Wedderburn, R. W. M. (1972). Generalized Linear Models.** *Journal of the Royal Statistical Society: Series A*, 135(3), 370–384. DOI: 10.2307/2344614
  * The seminal paper introducing the GLM framework, unifying regression models for exponential family distributions under a single estimation approach.

* **McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.).** Chapman and Hall. DOI: 10.1007/978-1-4899-3242-6
  * The definitive theoretical reference for GLMs. Covers distributional families, link functions, estimation via IRLS, and model diagnostics in depth.

* **Dobson, A. J., & Barnett, A. G. (2018). An Introduction to Generalized Linear Models (4th ed.).** Chapman and Hall/CRC. DOI: 10.1201/9781315182780
  * An accessible graduate-level introduction with applied examples across a range of families and link functions. Suitable for clinical and epidemiological researchers.

* **Harrell, F. E. Jr. (2015). Regression Modeling Strategies (2nd ed.).** Springer. DOI: 10.1007/978-3-319-19425-7
  * Covers variable selection, multicollinearity, splines for non-linear effects, and model validation strategies applicable to GLMs and related models.

* **McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.** In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
  * Introduces the McFadden pseudo R² as a likelihood-ratio-based measure of fit for models outside the OLS framework.

* **Akaike, H. (1974). A new look at the statistical model identification.** *IEEE Transactions on Automatic Control*, 19(6), 716–723. DOI: 10.1109/TAC.1974.1100705
  * Foundational paper introducing the Akaike Information Criterion (AIC) for model selection based on penalised log-likelihood.

* **Schwarz, G. (1978). Estimating the dimension of a model.** *Annals of Statistics*, 6(2), 461–464. DOI: 10.1214/aos/1176344136
  * Introduces the Bayesian Information Criterion (BIC), applying a stronger complexity penalty than AIC, particularly useful for large samples.

* **Cook, R. D. (1977). Detection of influential observation in linear regression.** *Technometrics*, 19(1), 15–18. DOI: 10.2307/1268249
  * Introduces Cook's distance as a measure of the influence of individual observations on regression estimates.

* **Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression, II.** *Biometrika*, 38(1–2), 159–177. DOI: 10.2307/2332325
  * Introduces the Durbin-Watson statistic for detecting first-order autocorrelation in regression residuals.

* **Rubin, D. B. (1976). Inference and missing data.** *Biometrika*, 63(3), 581–592. DOI: 10.2307/2335739
  * Establishes the MCAR / MAR / MNAR taxonomy for missing data mechanisms, which underpins the validity conditions for listwise deletion.