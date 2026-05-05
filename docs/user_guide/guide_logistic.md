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
    dependent_var="outcome",
    independent_vars=["age", "sex", "bmi"],
    regression_type="Multi"
)
```

### Parameters

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `data` | `pd.DataFrame` | required | The dataset to analyse. Must contain all outcome and predictor columns. Rows with missing values are dropped automatically. | Preprocessing |
| `dependent_var` | `str` | `None` | The name of the outcome (dependent) variable column. Must be binary and coded as 0/1. Required if `formula` is not provided. | Modeling |
| `independent_vars` | `list` | `None` | A list of predictor (independent) variable column names. Required if `formula` is not provided. | Modeling |
| `formula` | `str` | `None` | A Patsy-style formula string (e.g. `"outcome ~ age + sex"`). If provided, `dependent_var` and `independent_vars` are not required. | Modeling |
| `family` | `str` | `"binomial"` | The distributional family for the GLM. See [Supported Families and Links](#supported-families-and-links). | Modeling |
| `link` | `str` | `"logit"` | The link function for the GLM. See [Supported Families and Links](#supported-families-and-links). | Modeling |
| `regression_type` | `str` | `"Multi"` | Either `"Multi"` for multivariable regression or `"Uni"` for univariable regression. Affects column naming in the results table. | Modeling |
| `classification_threshold` | `float` | `0.5` | Probability threshold used to convert predicted probabilities into binary class predictions for classification metrics. | Modeling |

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

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `labels` | `dict` | `None` | A dictionary mapping raw variable names to human-readable labels for display in result tables. | Evaluation |
| `cross_val` | `bool` | `True` | Whether to perform k-fold cross-validation after fitting. | Validation |
| `n_splits` | `int` | `5` | Number of folds for cross-validation. Only used if `cross_val=True`. | Validation |

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

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `assumptions` | `str` or `list` | `None` | Pass `"all"` to show all assumption tests, or a list of specific test names. `None` skips this section. | Evaluation |
| `performance` | `str` or `list` | `None` | Pass `"all"` to show all performance metrics, or a list of specific metric names. `None` skips this section. | Evaluation |
| `cross_val` | `str` or `list` | `None` | Pass `"all"` to show all cross-validation metrics, or a list of specific metric names. `None` skips this section. | Evaluation |
| `plots` | `list` | `None` | A list of plot names to display. Options: `"forest_plot"`, `"roc_curve"`, `"confusion_matrix"`. | Evaluation |
| `vif_threshold` | `float` | `5.0` | The threshold above which a VIF value is flagged as indicating multicollinearity. | Evaluation |

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

## Statistical Notes

**1. The Logistic Regression Model**

Logistic regression models the probability of a binary outcome as a function of one or more predictors. It is a GLM with a Binomial family and a link function $g(\cdot)$ such that:

$$
g(\mu) = \eta = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
$$

where $\mu = P(Y = 1 \mid X)$. The default logit link expresses this as:

$$
\log \frac{P(Y=1 \mid X)}{1 - P(Y=1 \mid X)} = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
$$

The left-hand side is the log-odds of the event. Parameters are estimated by maximum likelihood. Unlike linear regression, there is no closed-form solution; estimation proceeds iteratively via IRLS.

---

**Interpreting the Model**

**1. Odds Ratio (OR)**

The odds ratio quantifies the effect of a predictor on the odds of the event occurring. It is calculated as:

$$
\text{OR} = e^{\beta_j}
$$

where:

*   If $\text{OR} > 1$, the predictor increases the odds of the event.
*   If $\text{OR} < 1$, the predictor decreases the odds of the event.
*   If $\text{OR} = 1$, the predictor has no effect on the odds.

For a one-unit increase in $X_j$, the odds of the event are multiplied by $e^{\beta_j}$, holding all other predictors constant.

**2. Confidence Intervals (CI)**

The 95% confidence interval for the odds ratio is:

$$
\text{CI} = \left[ e^{\beta_j - 1.96 \cdot \text{SE}(\beta_j)},\ e^{\beta_j + 1.96 \cdot \text{SE}(\beta_j)} \right]
$$

where $\text{SE}(\beta_j)$ is derived from the Fisher information matrix.

*   If the CI excludes 1, the result suggests a statistically significant association between the predictor and the outcome.
*   If the CI includes 1, the data do not provide strong enough evidence to conclude that an association exists.

**3. p-value**

The p-value tests whether the predictor has a significant effect on the outcome:

*   If $p < 0.05$, the variable is statistically significant.
*   If $p > 0.05$, there is no strong evidence that the predictor affects the outcome.

**4. Alternative link functions**

The probit and complementary log-log (cloglog) links produce coefficients that are not odds ratios and require different interpretation.

Under the **probit link**, the coefficient represents the change in the standard normal quantile of $P(Y=1)$ per unit increase in $X_j$. Probit coefficients can be approximately converted to log-odds by multiplying by $\pi / \sqrt{3} \approx 1.81$, but direct interpretation as odds ratios is not valid.

Under the **cloglog link**, the model is:

$$
\log(-\log(1 - P(Y=1 \mid X))) = \eta
$$

The exponentiated coefficient approximates a **log hazard ratio** rather than an odds ratio, and is most appropriate when the binary outcome derives from an underlying continuous time-to-event process observed at a single time point, or when the event probability is very low. Interpreting cloglog coefficients as odds ratios is incorrect.

---

**Performance Metrics**

*   **AUC-ROC**: The area under the receiver operating characteristic curve measures the model's ability to discriminate between events and non-events across all possible classification thresholds. A value of 0.5 indicates no discrimination (equivalent to chance); a value of 1.0 indicates perfect discrimination. AUC is threshold-independent and is the primary discrimination metric for logistic regression.

*   **Accuracy, Precision, Recall, F1**: Classification metrics computed at the specified `classification_threshold` (default 0.5). These are threshold-dependent and sensitive to class imbalance. In datasets where events are rare, a model that predicts 0 for every observation can achieve high accuracy while having no discriminative value. AUC-ROC should be preferred for imbalanced outcomes.

*   **Cross-validation accuracy**: The pipeline cross-validates using accuracy as its sole CV metric. This inherits the same class imbalance problem as in-sample accuracy — high CV accuracy on an imbalanced dataset does not indicate a useful model. AUC-ROC is not cross-validated by this pipeline. The fold split uses a fixed random seed (`random_state=42`); results are reproducible but analysts requiring seed sensitivity analysis should implement CV externally.

*   **Pseudo R²**: Several pseudo R² statistics are reported as approximate analogues to OLS R². They are not directly comparable to each other or to standard R², and should not be interpreted as the proportion of variance explained. McFadden R² values between 0.2 and 0.4 are generally considered good fit. Tjur R² (the difference in mean predicted probabilities between events and non-events) is often the most intuitive for binary outcomes.

*   **AIC and BIC**: Penalised log-likelihood metrics used for comparing models fit to the same data. Lower values indicate better fit relative to the number of parameters. BIC applies a stronger complexity penalty and is preferred when the goal is identifying the true model.

---

**Advantages**

*   Produces directly interpretable odds ratios with confidence intervals for each predictor.
*   Handles binary outcomes without requiring distributional assumptions about the predictors.
*   AUC-ROC provides a threshold-independent measure of model discrimination, appropriate for imbalanced outcomes.
*   Multiple link functions (logit, probit, cloglog) accommodate different assumptions about the underlying event probability structure.

---

**Limitations**

*   The outcome variable must be coded strictly as `0` and `1` before passing data to the pipeline. Boolean columns, factor-coded variables (e.g. 1/2), or string labels will raise a validation error. Recoding must be applied upstream.
*   Odds ratios overestimate relative risk when the outcome is common (prevalence above approximately 10%). In high-prevalence settings, relative risk regression (log-binomial or Poisson with robust standard errors) should be considered.
*   The model is sensitive to complete separation, where a predictor or combination of predictors perfectly predicts the outcome. This causes the maximum likelihood estimator to diverge; coefficients and standard errors become unreliable or infinite.
*   Listwise deletion is applied automatically for missing values. This is valid only under Missing Completely At Random (MCAR). Multiple imputation should be applied upstream for datasets with substantial missingness.
*   The `classification_threshold` default of 0.5 is arbitrary and may be inappropriate when the costs of false positives and false negatives differ, or when the outcome is rare.

---

**Assumptions of the Logistic Regression Model**

The logistic regression model relies on several critical assumptions that must hold for its estimates to be valid and interpretable.

<br>

* **Linearity on the Log-Odds Scale**: The model assumes a linear relationship between each continuous predictor and the log-odds of the outcome. This does not apply to categorical variables, which are handled through dummy encoding.
  * **Implications**: Non-linearity leads to biased odds ratio estimates and reduced model fit.
  * **If Violated**: Use polynomial terms or splines (e.g. restricted cubic splines) to better capture the functional form.
  * **Evaluation**: Plot the log-odds of the outcome against each continuous predictor. Non-linear patterns indicate a violation. The Box-Tidwell test can be used as a formal assessment.

<br>

* **Independence of Observations**: Observations must be independent of one another. This assumption may be violated in clustered or repeated-measures designs.
  * **Implications**: Ignoring dependence leads to underestimated standard errors and anti-conservative p-values. Odds ratio estimates remain approximately unbiased but inference is invalid.
  * **If Violated**: Use mixed-effects logistic regression (random effects) or apply cluster-robust standard errors.
  * **Evaluation**: Not directly testable from the data alone. Review study design for clustering (e.g. patients nested within hospitals) and examine residual structure across groups.

<br>

* **No Influential Outliers**: No individual observation exerts disproportionate leverage on the coefficient estimates.
  * **Implications**: Highly influential observations can substantially shift estimated odds ratios, producing results that do not generalise.
  * **If Violated**: Investigate flagged observations before removing them. Report analyses with and without the influential point.
  * **Evaluation**: Cook's distance threshold of $4/n$ is applied (`model.influential_outliers_threshold`). This threshold becomes permissive for large $n$; graphical inspection of the influence plot is advisable for large datasets.

<br>

* **No Multicollinearity**: Predictor variables are not strongly linearly related to one another.
  * **Implications**: High collinearity inflates standard errors, making individual odds ratio estimates unstable and difficult to interpret. Overall model discrimination (AUC) is unaffected.
  * **If Violated**: Remove one of the correlated predictors, combine them into a composite, or use penalised regression (ridge or lasso logistic regression).
  * **Evaluation**: Variance Inflation Factors are computed for each predictor (`model.vif_df`). The default flag threshold is VIF > 5.0, a widely used heuristic; some fields use 10. The threshold is configurable via `vif_threshold`.

<br>

* **Sufficient Events Per Variable (EPV)**: The number of outcome events relative to the number of predictors must be adequate for stable estimation.
  * **Implications**: Too few events per variable produces biased and unstable coefficient estimates, inflated standard errors, and overfitting. A commonly cited minimum is 10 EPV, though recent simulation evidence suggests as few as 5 may be acceptable under certain conditions.
  * **If Violated**: Reduce the number of predictors, apply penalised regression to regularise estimates, or use shrinkage methods such as the Firth correction.
  * **Evaluation**: EPV is reported automatically (`model.epv`). Values below 10 are flagged as potentially problematic.

<br>

* **Missing Completely At Random**: Rows with missing values are dropped automatically. This is valid only if the probability of missingness is unrelated to any observed or unobserved variable.
  * **Implications**: If data are Missing At Random (MAR) or Missing Not At Random (MNAR), listwise deletion produces biased odds ratio estimates whose severity depends on the proportion missing and the mechanism.
  * **If Violated**: Apply multiple imputation (e.g. MICE) before passing data to the pipeline.
  * **Evaluation**: Compare the distribution of key covariates in complete vs. incomplete cases. Systematic differences suggest non-MCAR missingness.

<br>

---

### References

* **Cox, D. R. (1958). The regression analysis of binary sequences.** *Journal of the Royal Statistical Society: Series B*, 20(2), 215–242. DOI: 10.1111/j.2517-6161.1958.tb00292.x
  * The foundational paper introducing logistic regression for binary outcomes.

* **Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.).** Wiley. DOI: 10.1002/9781118548387
  * The standard applied reference for logistic regression in clinical and epidemiological research. Covers interpretation, model building, and diagnostics in depth.

* **Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1996). A simulation study of the number of events per variable in logistic regression analysis.** *Journal of Clinical Epidemiology*, 49(12), 1373–1379. DOI: 10.1016/S0895-4356(96)00236-3
  * Classic simulation study establishing the practical rule of 10 events per variable (EPV) for stable logistic regression estimates.

* **Vittinghoff, E., & McCulloch, C. E. (2007). Relaxing the rule of ten events per variable in logistic and Cox regression.** *American Journal of Epidemiology*, 165(6), 710–718. DOI: 10.1093/aje/kwk052
  * Simulation evidence that as few as 5–9 EPV can produce reliable estimates under controlled conditions, particularly with regularisation.

* **Firth, D. (1993). Bias reduction of maximum likelihood estimates.** *Biometrika*, 80(1), 27–38. DOI: 10.2307/2336755
  * Introduces the Firth penalised likelihood correction for logistic regression, the standard approach to handling complete or quasi-complete separation.

* **Tjur, T. (2009). Coefficients of determination in logistic regression models — a new proposal.** *The American Statistician*, 63(4), 366–372. DOI: 10.1198/tast.2009.08210
  * Introduces Tjur's R² as a simple, interpretable discrimination measure for binary outcomes based on the difference in mean predicted probabilities between outcome groups.

* **McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.** In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
  * Introduces the McFadden pseudo R² as a likelihood-ratio-based measure of fit for non-OLS models.

* **Harrell, F. E. Jr. (2015). Regression Modeling Strategies (2nd ed.).** Springer. DOI: 10.1007/978-3-319-19425-7
  * Covers variable selection, EPV, splines, penalised regression, and validation strategies for logistic models. Includes the Firth correction and calibration assessment.

* **Steyerberg, E. W. (2019). Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating (2nd ed.).** Springer. DOI: 10.1007/978-3-030-16399-0
  * Comprehensive guide to logistic model development and validation including calibration, discrimination, and internal vs. external validation.

* **Rubin, D. B. (1976). Inference and missing data.** *Biometrika*, 63(3), 581–592. DOI: 10.2307/2335739
  * Establishes the MCAR / MAR / MNAR taxonomy for missing data mechanisms underpinning the validity conditions for listwise deletion.