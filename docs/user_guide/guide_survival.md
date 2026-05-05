# `Survival Analysis (Cox)`

`class survival_pipeline.RAPID_SurvivalCox(data, duration_var, dependent_var, independent_vars)` [[source]](https://github.com/noispuc/isaric3.0-rapid/blob/main/src/isaric/pipelines/)

> Survival analysis is a branch of statistics dedicated to analysing the time until one or more events occur (time-to-event). The Cox Proportional Hazards Model is a semi-parametric regression model used to assess the effect of predictor variables on survival time. It does not require specifying the underlying distribution of survival times, only that the hazard ratio between any two individuals remains constant over time.

---

The `RAPID_SurvivalCox` class is a specialized pipeline designed for **Survival Analysis** within epidemiological contexts. It facilitates the end-to-end workflow from raw data to the estimation of Cox Proportional Hazards models and the generation of clinical reports.

---

## Initialisation

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

pipeline = factory.create(
    "survival",
    data=clinical_df,
    duration_var='days_to_event',
    dependent_var='outcome_death',
    independent_vars=['age', 'sex', 'bmi', 'comorbidity_index']
)
```

### Parameters

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `data` | `pd.DataFrame` | required | The input dataset containing the variables for analysis. | Preprocessing |
| `duration_var` | `str` | required | The name of the column representing time-to-event (e.g., days until discharge or death). | Modeling |
| `dependent_var` | `str` | required | The name of the binary column where `1` indicates the event occurred and `0` indicates censoring. | Modeling |
| `independent_vars` | `list of str` | required | A list of feature names to be used as covariates in the survival model. | Modeling |

---

## fit()

Fits the Cox Proportional Hazards model and runs all evaluation steps.

```python
pipeline.fit(
    labels={'age': 'Age (years)', 'bmi': 'Body Mass Index'},
    penalizer=0.05,
    cross_val=True,
    n_splits=5
)
```

### Parameters

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `formula` | `str` | `None` | R-style formula string for data transformation. Implemented with the `Formulaic` library. Allows specifying interactions and custom model structures. See [Formulaic docs](https://matthewwardrop.github.io/formulaic/latest/). | Modeling |
| `labels` | `dict` | `None` | A dictionary mapping internal column names to human-readable labels for reporting. | Evaluation |
| `penalizer` | `float` | `0.1` | L2 regularization parameter to improve model stability and prevent overfitting. | Modeling |

---

## summary()

Displays results after fitting. All arguments are optional — pass only what you want to see.

```python
pipeline.summary(
    assumptions=True,
    performance=True,
    plots=['forest_plot', 'roc_auc', 'martingale'],
    target_time=28
)
```

### Parameters

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `assumptions` | `bool` | `False` | If `True`, displays results of assumption tests including Multicollinearity (VIF) and Influential Outliers (Cook's Distance, Leverage, DFBetas). | Evaluation |
| `performance` | `bool` | `False` | If `True`, displays performance metrics including Accuracy, Precision, Recall, F1 Score, Log Loss, and ROC AUC Score. | Evaluation |
| `plots` | `list of str` | `None` | A list of plot names to display. Supported options: `"forest_plot"`, `"roc_auc"`, `"martingale"`. | Evaluation |
| `target_time` | `float` | `None` | The specific time point used for calculating ROC curves and AUC. | Evaluation |

---

## Examples

### Standard implementation

```python
from isaric.pipelines.factory import RAPID_PipelineFactory

factory = RAPID_PipelineFactory()

pipeline = factory.create(
    "survival",
    data=clinical_df,
    duration_var='days_to_event',
    dependent_var='outcome_death',
    independent_vars=['age', 'sex', 'bmi', 'comorbidity_index']
)

pipeline.fit(
    labels={'age': 'Age (years)', 'bmi': 'Body Mass Index'},
    penalizer=0.05,
    cross_val=True,
    n_splits=5
)

pipeline.summary(
    performance=True,
    assumptions=True,
    plots=['forest_plot', 'roc_auc'],
    target_time=28
)
```

### Using a custom formula

```python
pipeline_c2 = factory.create(
    "survival",
    data=df_map,
    duration_var='duration_var',
    dependent_var='outcome_binary',
    independent_vars=['demog_sex', 'comor_hypertensi', 'comor_obesity']
)

custom_formula = "duration_var + outcome_binary ~ demog_sex * comor_obesity + comor_hypertensi"

pipeline_c2.fit(formula=custom_formula, penalizer=0.1)

pipeline_c2.summary(
    performance=True,
    assumptions=True,
    plots=['martingale']
)
```

---

## Development Notes

> [!IMPORTANT]
>
> In this version, survival analysis is conducted solely using the listwise deletion method to address missing data.
>
> To implement the Cox Proportional Hazards model we used the `lifelines` library — a specialized package for survival analysis. It provides easy-to-use tools for fitting and interpreting models like Kaplan-Meier, Cox Proportional Hazards, and more. You can learn more about its official documentation at: https://lifelines.readthedocs.io
>
> To implement the formulas for personalized X and y matrices generation we used `Formulaic`. A high-performance implementation of Wilkinson formulas for Python, which are very useful for transforming dataframes into a form suitable for ingestion into various modelling frameworks. You can learn more about its official documentation at: https://matthewwardrop.github.io/formulaic/latest/

---

## Statistical Notes

**The Hazard Function in the Cox Model**

The hazard function, denoted as $h(t)$, represents the instantaneous risk of an event occurring at time $t$, given survival up to that time. The Cox model assumes that this hazard function can be expressed as:

$$
h(t | X) = h_0(t) e^{(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p)}
$$

where:

- $h_0(t)$ is the baseline hazard function, representing the risk when all predictor variables are zero.
- $X_1, X_2, ..., X_p$ are the predictor variables (covariates).
- $\beta_1, \beta_2, ..., \beta_p$ are the coefficients that measure the impact of each predictor on survival.

This formulation allows us to analyze the effect of covariates on survival without making assumptions about the baseline hazard $h_0(t)$.

**Proportional Hazards Assumption**

The term proportional hazards comes from the assumption that the hazard ratios between individuals remain constant over time. That is, the effect of a covariate does not change as time progresses. Mathematically, for two individuals with predictor values $x_A$ and $x_B$:

$$
\frac{h(t | X = x_A)}{h(t | X = x_B)} = e^{(\beta_1 (x_{A1} - x_{B1}) + ... + \beta_p (x_{Ap} - x_{Bp}))}
$$

Since $h_0(t)$ cancels out, the hazard ratio is independent of time $t$. If this assumption does not hold, alternative models like time-dependent covariates or stratified Cox models may be necessary.

---

## Interpreting the Cox Model

**Hazard Ratio ($\mathrm{HR}$)**

The hazard ratio quantifies the effect of a predictor variable on survival:

$$
\mathrm{HR} = e^{\beta}
$$

- If $\mathrm{HR} > 1$, the predictor increases the hazard (higher risk, shorter survival time).
- If $\mathrm{HR} < 1$, the predictor decreases the hazard (lower risk, longer survival time).
- If $\mathrm{HR} = 1$, the predictor has no effect on survival.

**Confidence Intervals ($\mathrm{CI}$)**

To assess statistical significance, we compute the 95% confidence interval for the hazard ratio:

$$
\mathrm{CI} = \left[ e^{(\beta - 1.96 \cdot \sigma)}, e^{(\beta + 1.96 \cdot \sigma)} \right]
$$

where $\sigma$ is the standard error of the coefficient.

- If the $\mathrm{CI}$ excludes 1, the result suggests a statistically significant association between the predictor and the hazard.
- If the $\mathrm{CI}$ includes 1, the data do not provide strong enough evidence to conclude that an association exists.

**p-value**

The p-value tests whether the predictor has a significant effect on survival:

- If $p < 0.05$, the variable is statistically significant.
- If $p > 0.05$, there is no strong evidence that the predictor affects survival.

---

## Advantages

- **No assumption on survival time distribution**: Unlike parametric models, the Cox model does not require specifying the shape of the survival curve.
- **Handles censored data well**: It efficiently includes individuals for whom the event has not yet occurred.
- **Interpretable coefficients**: The exponentiated coefficients provide direct insights into risk factors.

---

## Limitations

- **Time-dependent covariates**: While the basic Cox model assumes time-independent covariates, it can be extended to handle time-dependent covariates using appropriate data structures and modeling techniques.
- **Baseline hazard is not estimated**: The model focuses on relative risks rather than predicting absolute survival probabilities.

---

## Assumptions of the Cox Proportional Hazards Model

The Cox model relies on several critical assumptions that must hold for its estimates to be valid and interpretable. These assumptions relate to the nature of covariate effects, data structure, and censoring mechanisms.

<br>

- **Proportional Hazards**: The hazard ratios between individuals are constant over time. The effect of a covariate is multiplicative with respect to the baseline hazard and does not vary during follow-up.
  - **Implications**: Enables a time-invariant interpretation of covariate effects as hazard ratios.
  - **If Violated**: Incorporate time-dependent covariate interactions or use stratified Cox models.
  - **Evaluation**: Checked using Schoenfeld residuals or log(-log(survival)) vs. log(time) plots. Temporal trends indicate violations.

<br>

- **Linearity on the Log-Hazard Scale**: The model assumes a linear relationship between continuous covariates and the logarithm of the hazard function. This does not apply to categorical variables, which are handled through dummy encoding.
  - **Implications**: Non-linearity may lead to biased effect estimates and reduced model fit.
  - **If Violated**: Use polynomial terms or splines (e.g., restricted cubic splines) to better model the functional form.
  - **Evaluation**: Martingale residuals plotted against covariates can reveal non-linear patterns.

<br>

- **Independence of Observations**: Survival times across individuals must be independent. This assumption may be violated in clustered or repeated-measures designs.
  - **Implications**: Ignoring dependence leads to underestimated standard errors and invalid inferences.
  - **If Violated**: Use frailty models or apply robust (clustered) standard errors.
  - **Evaluation**: Not directly testable; however, study design review and examining residual structure across groups (e.g., hospitals) may reveal dependence.

<br>

- **Non-Informative Censoring**: The probability of being censored must be independent of the underlying event risk, conditional on covariates.
  - **Implications**: Informative censoring can bias hazard ratios and survival estimates.
  - **If Violated**: Consider joint models or alternative frameworks that explicitly model the censoring process.
  - **Evaluation**: This cannot be formally tested with survival data alone. Assess whether censored individuals differ systematically from those with observed events (e.g., via descriptive statistics or survival curves stratified by censoring groups).

---

## References

* **Cox, D. R. (1972). Regression models and life-tables**. *Journal of the Royal Statistical Society: Series B (Methodological)*, 34(2), 187–220. DOI: 10.1111/j.2517-6161.1972.tb00899.x
  * This seminal paper introduced the proportional hazards model, laying the theoretical foundation for what became the Cox regression model. It remains the cornerstone reference for understanding hazard functions and survival regression.

* **Hosmer, D. W., Lemeshow, S., & May, S. (2008). Applied Survival Analysis: Regression Modeling of Time-to-Event Data** (2nd ed.). Wiley-Interscience. DOI: 10.1002/0471754994
  * A widely used reference for practical application of Cox models in healthcare and clinical research. Includes extensive examples using statistical software such as SAS and SPSS.

* **Therneau, T. M., & Grambsch, P. M. (2000). Modelling Survival Data: Extending the Cox Model**. Springer. DOI: 10.1007/978-1-4757-3294-8
  * This book is considered essential for understanding extensions of the Cox model, including time-varying covariates and stratified models. Written by the developer of the *survival* package in R. Additionally, it is also a fundamental book on the Cox covering key assumptions (proportional hazards, independence, non-informative censoring).

* **Collett, D. (2023). Modelling Survival Data in Medical Research** (4th ed.). Chapman and Hall/CRC. DOI: 10.1201/9781003282525
  * Covers both the Kaplan-Meier and Cox models with clarity and depth, using real-life medical examples and datasets. Suitable for graduate students and professionals.

* **Kleinbaum, D. G., & Klein, M. (2012). Survival Analysis: A Self-Learning Text** (3rd ed.). Springer. DOI: 10.1007/978-1-4419-6646-9
  * A reader-friendly introduction to survival models including Cox regression. Features intuitive explanations and worked examples using software like R and Stata.

* **Tibshirani, R. (1997). The Lasso Method for Variable Selection in the Cox Model**. *Statistics in Medicine*.
  * Introduces the LASSO method for automatic variable selection in Cox regression by penalizing less important coefficients.

* **Lin, Y., & Zhang, H. H. (2006). Component selection and smoothing in smoothing spline analysis of variance models**. *Annals of Statistics*, 34(5), 2272–2297. DOI: 10.1214/009053606000000604
  * This methodological paper introduces the COSSO framework, a unified approach for variable selection and function estimation in nonparametric regression models. Though not specific to survival analysis, later extensions have adapted COSSO to Cox models, enabling flexible modeling of non-linear covariate effects with automatic selection.

* **Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1995). A Simulation Study of the Number of Events per Variable in Logistic Regression Analysis**. *Journal of Clinical Epidemiology*.
  * Classic study that established the practical rule of 10 events per variable (EPV) to ensure estimate stability.

* **Vittinghoff, E., & McCulloch, C. E. (2007). Relaxing the Rule of Ten Events per Variable in Logistic and Cox Regression**. *American Journal of Epidemiology*, 165(6), 710–718. DOI: 10.1093/aje/kwk052
  * This study uses simulation methods to show that Cox regression models with as few as 5–9 events per variable (EPV) can produce reliable estimates, especially when effects are strong or model complexity is controlled with techniques like regularization or bootstrapping.

* **Harrell, F. E. Jr. (2015). Regression Modeling Strategies**. Springer.
  * Covers variable selection, handling multicollinearity, use of splines for nonlinear effects, and strategies to deal with censoring.

* **Steyerberg, E. W. (2019). Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating** (2nd ed.). Springer. DOI: 10.1007/978-3-030-16399-0
  * Comprehensive guide to model development and validation, including internal and external validation techniques, calibration methods such as calibration belts, and performance metrics like the C-index and Brier score. Widely used in clinical research and survival modeling.

* **Gönen, M., & Heller, G. (2005). Concordance probability and discriminatory power in proportional hazards regression**. *Biometrika*, 92(4), 965–970. DOI: 10.1093/biomet/92.4.965
  * Foundational paper on the mathematical definition and interpretation of the concordance index (C-index) in proportional hazards models. Provides a solid theoretical basis for discrimination assessment in survival analysis.

* **Royston, P., & Altman, D. G. (2010). Visualizing and assessing discrimination in the logistic regression model using the C-statistic**. *Statistics in Medicine*, 29(24), 2506–2516. DOI: 10.1002/sim.3994
  * Discusses visualization and interpretation of the C-statistic, offering insights into model discrimination. While focused on logistic models, its principles are often applied to C-index interpretation in survival analysis.

* **Heagerty, P. J., & Zheng, Y. (2005). Survival model predictive accuracy and ROC curves**. *Biometrics*, 61(1), 92–105. DOI: 10.1111/j.0006-341X.2005.030814.x
  * Introduces the concept of time-dependent ROC curves and AUC for censored survival data. Offers formal methods to evaluate predictive accuracy of Cox models at specific time points. A key reference for dynamic discrimination measures in survival analysis.