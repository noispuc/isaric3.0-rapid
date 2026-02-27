# `survival_cox.RAPID_SurvivalCox`

`class survival_pipeline.RAPID_SurvivalCox(data, duration_var, dependent_var, independent_vars)` [[source]](https://github.com/noispuc/isaric3.0-rapid/blob/main/src/isaric/pipelines/)

> Suvival analysis is a branch of statistics dedicated to analysing the time until one or more events occur (time-to-event). The Cox Proportional Hazards Model is a semi-parametric regression model used to assess the effect of predictor variables on survival time. It does not require specifying the underlying distribution of survival times, only that the hazard ratio between any two individuals remains constant over time. 


---

The `RAPID_SurvivalCox` class is a specialized pipeline designed for **Survival Analysis** within epidemiological contexts. It facilitates the end-to-end workflow from raw data to the estimation of Cox Proportional Hazards models and the generation of clinical reports.



### Parameters

* **data** (*pd.DataFrame*):
The input dataset containing the variables for analysis.
* **duration_var** (*str*):
The name of the column representing time-to-event (e.g., days until discharge or death).
* **dependent_var** (*str*):
The name of the binary column where `1` indicates the event occurred and `0` indicates censoring.
* **independent_vars** (*list of str*):
A list of feature names to be used as covariates in the survival model.

---

### Methods

#### `fit(formula=None, labels=None, penalizer=0.1)`

Executes the full modeling sequence.

* **Parameters:**
* **formula** (*str, optional*):  R-style formula string from lists of column names for data transformation using formulas.

    * Formulas are useful because they provide a concise and explicit specification for how data should be prepared for a model.

    * The formulas for personalized X and y matrices generation were implemented with <code>Formulaic</code> library. You can learn more about it's own official documentation at: https://matthewwardrop.github.io/formulaic/latest/

* **labels** (*dict, optional*): A dictionary mapping internal column names to "pretty" labels for reporting.
* **penalizer** (*float, default=0.1*): L2 regularization parameter to improve model stability and prevent overfitting.



#### `summary(assumptions=False, performance=True, plots=None, target_time=None)`

Reports the Cox model findings and triggers visualization tools.

* **Parameters:**
 * **assumptions** (*bool, default=False*): If `True`, displays results of assumption tests including:
    * Multicollinearity (VIF)
    * Influential Outliers (Cook's Distance, Leverage, DFBetas)
 * **performance** (*bool, default=False*): If `True`, displays classification performance metrics including:
    * Accuracy
    * Precision
    * Recall
    * F1 Score
    * Log Loss
    * ROC AUC Score
* **plots** (*list of str, optional*): Types of plots to generate. Supported options:
    * `['forest_plot', 'roc_auc', 'martingale']`.
* **target_time** (*float, optional*): The specific time point used for calculating ROC curves and AUC.


### Examples

Below is a standard implementation of the survival pipeline:

```python
from survival_cox import RAPID_SurvivalCox

# 1. Initialize the pipeline
pipeline = RAPID_SurvivalCox(
    data=clinical_df,
    duration_var='days_to_event',
    dependent_var='outcome_death',
    independent_vars=['age', 'sex', 'bmi', 'comorbidity_index']
)

# 2. Fit the model with specific labels for the output
pipeline.fit(
    labels={'age': 'Age (years)', 'bmi': 'Body Mass Index'},
    penalizer=0.05
)

# 3. Generate summary and Forest Plot
pipeline.summary(
    performance=True,
    assumptions = True,
    plots=['forest_plot', 'roc_auc'],
    target_time=28)

```

Here is a similar example but with the use of formulas

```python
# Instantiation with processed df_map
pipeline_c2 = RAPID_SurvivalCox(
    data=df_map,
    duration_var='duration_var',
    dependent_var='outcome_binary',
    independent_vars=['demog_sex', 'comor_hypertensi', 'comor_obesity']
)

# 3. Fit using a Custom Formula
# This allows testing interactions like Sex * Obesity
custom_formula = "duration_var + outcome_binary ~ demog_sex * comor_obesity + comor_hypertensi"

print("Fitting Model Case 2 with Formula...")
pipeline_c2.fit(formula=custom_formula, penalizer=0.1)

# 4. Summary with Martingale Residuals
# Useful for checking linearity of continuous independent_vars
pipeline_c2.summary(
    performance=True,
    assumptions=True,
    plots=['martingale']
)

```
### Development notes

> [! IMPORTANT]

> In this version, survival analysis is conducted solely using the listwise deletion method to address missing data.

> To implement the Cox Proportional we used the <code>lifelines</code> library — a specialized package for survival analysis. It provides easy-to-use tools for fitting and interpreting models like Kaplan-Meier, Cox Proportional Hazards, and more. You can learn more about it's own official documentation at:
https://lifelines.readthedocs.io

> To implement the formulas for personalized X and y matrices generation we used <code>Formulaic</code>. A high-performance implementation of Wilkinson formulas for Python, which are very useful for transforming dataframes into a form suitable for ingestion into various modelling frameworks. You can learn more about it's own official documentation at:
https://matthewwardrop.github.io/formulaic/latest/
---
### Statistical Notes
**1. Hazard Function in the Cox Model**

<dd>

The hazard function, denoted as $h(t)$, represents the instantaneous risk of an event occurring at time $t$, given survival up to that time. The Cox model assumes that this hazard function can be expressed as:

$$
h(t | X) = h_0(t) e^{(\beta_1 X_1 + \beta_2 X_2 + ... + \beta_p X_p)}
$$

where:


*   $h_0(t)$ is the baseline hazard function, representing the risk when all predictor variables are zero.
*   $X_1, X_2, ..., X_p$ are the predictor variables (covariates).
*   $\beta_1, \beta_2, ..., \beta_p$ are the coefficients that measure the impact of each predictor on survival.


This formulation allows us to analyze the effect of covariates on survival without making assumptions about the baseline hazard $h_0(t)$.

<dt>

**2. Proportional Hazards Assumption**

<dd>

The term proportional hazards comes from the assumption that the hazard ratios between individuals remain constant over time. That is, the effect of a covariate does not change as time progresses. Mathematically, for two individuals with predictor values $x_A$ and $x_B$:

$$
\frac{h(t | X = xA)}{h(t | X = xB)} = e^{(\beta_1 (x_{A1} - x_{B1}) + ... + \beta_p (x_{Ap} - x_{Bp}))}
$$

Since $h_0(t)$ cancels out, the hazard ratio is independent of time $t$.

If this assumption does not hold, alternative models like time-dependent covariates or stratified Cox models may be necessary.

<dt>

<br>

---

<br>

**Interpreting the Cox Model**

**1. Hazard Ratio ($\mathrm{HR}$)**

<dd>

The hazard ratio (HR) quantifies the effect of a predictor variable on survival. It is calculated as:

$$
\mathrm{HR} = e^{\beta}
$$

where:


*   If $\mathrm{HR} > 1$, the predictor increases the hazard (higher risk, shorter survival time).

*   If $\mathrm{HR} < 1$, the predictor decreases the hazard (lower risk, longer survival time).

*   If $\mathrm{HR} = 1$, the predictor has no effect on survival.


<dt>

**2. Confidence Intervals ($\mathrm{CI}$)**

<dd>

To assess statistical significance, we compute the 95% confidence interval (CI) for the hazard ratio:

$$
\mathrm{CI} = \left[ e^{(\beta - 1.96 \cdot  σ)}, e^{(\beta + 1.96 \cdot  σ)} \right]
$$

where ${σ}$ is the standard error of the coefficient.

*   If the $\mathrm{CI}$ excludes 1, the result suggests a statistically significant association between the predictor and the hazard.
*   If the $\mathrm{CI}$ includes 1, the data do not provide strong enough evidence to conclude that an association between the predictor and the hazard exists.


<dt>

**3. p-value**

<dd>

The p-value tests whether the predictor has a significant effect on survival:

*   If p < 0.05, the variable is statistically significant.
*   If p > 0.05, there is no strong evidence that the predictor affects survival.

<dt>

<br>

---

<br>

**Advantages**


*   No assumption on survival time distribution: Unlike parametric models, the Cox model does not require specifying the shape of the survival curve.

*   Handles censored data well: It efficiently includes individuals for whom the event has not yet occurred.

*   Interpretable coefficients: The exponentiated coefficients provide direct insights into risk factors.

<br>

---

<br>


**Limitations**

*   Time-dependent covariates: While the basic Cox model assumes time-independent covariates, it can be extended to handle time-dependent covariates using appropriate data structures and modeling techniques.

*   Baseline hazard is not estimated: The model focuses on relative risks rather than predicting absolute survival probabilities.

<br>

---

<br>

**Assumptions of the Cox Proportional Hazards Model**

<dd>

The Cox model relies on several critical assumptions that must hold for its estimates to be valid and interpretable. These assumptions relate to the nature of covariate effects, data structure, and censoring mechanisms.

<br>

* **Proportional Hazards Assumption**: This is the primary structural assumption of the Cox model. It states that the hazard ratios between individuals are constant over time. That is, the effect of a covariate is multiplicative with respect to the baseline hazard and does not vary during follow-up.
  * **Implications**: Enables a time-invariant interpretation of covariate effects as hazard ratios.
  * **If Violated**: Incorporate time-dependent covariate interactions or use stratified Cox models.
  * **Evaluation**: Checked using Schoenfeld residuals or log(-log(survival)) vs. log(time) plots. Temporal trends indicate violations.

<br>

* **Linearity on the Log-Hazard Scale**: The model assumes a linear relationship between continuous covariates and the logarithm of the hazard function. This does not apply to categorical variables, which are handled through dummy encoding.
  * **Implications**: Non-linearity may lead to biased effect estimates and reduced model fit.
  * **If Violated**: Use polynomial terms or splines (e.g., restricted cubic splines) to better model the functional form.
  * **Evaluation**: Martingale residuals plotted against covariates can reveal non-linear patterns.

<br>

* **Independence of Observations**: Survival times across individuals must be independent. This assumption may be violated in clustered or repeated-measures designs.
  * **Implications**: Ignoring dependence leads to underestimated standard errors and invalid inferences.
  * **If Violated**: Use frailty models or apply robust (clustered) standard errors.
  * **Evaluation**: Not directly testable; however, study design review and examining residual structure across groups (e.g., hospitals) may reveal dependence.

<br>

* **Non-Informative Censoring**: The probability of being censored must be independent of the underlying event risk, conditional on covariates.
  * **Implications**: Informative censoring can bias hazard ratios and survival estimates.
  * **If Violated**: Consider joint models or alternative frameworks that explicitly model the censoring process.
  * **Evaluation**: This cannot be formally tested with survival data alone. Assess whether censored individuals differ systematically from those with observed events (e.g., via descriptive statistics or survival curves stratified by censoring groups).

<dt>

<br>




### **References**


If you wish to further explore the Cox Proportional Hazards model—its assumptions, estimation methods, applications in medical research, and extensions—there are several foundational and practical works that provide in-depth explanations. Below are some of the most recommended references for both theoretical and applied studies of the method:


* **Cox, D. R. (1972). Regression models and life-tables**. Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187–220. DOI: 10.1111/j.2517-6161.1972.tb00899.x
    * This seminal paper introduced the proportional hazards model, laying the theoretical foundation for what became the Cox regression model. It remains the cornerstone reference for understanding hazard functions and survival regression.

* **Hosmer, D. W., Lemeshow, S., & May, S. (2008). Applied Survival Analysis: Regression Modeling of Time-to-Event Data** (2nd ed.). Wiley-Interscience. DOI: 10.1002/0471754994
    * A widely used reference for practical application of Cox models in healthcare and clinical research. Includes extensive examples using statistical software such as SAS and SPSS.

* **Therneau, T. M., & Grambsch, P. M. (2000). Modelling Survival Data: Extending the Cox Model. Springer**. DOI: 10.1007/978-1-4757-3294-8
    * This book is considered essential for understanding extensions of the Cox model, including time-varying covariates and stratified models. Written by the developer of the *survival* package in R. Additionally, it is also a fundamental book on the Cox covering key assumptions (proportional hazards, independence, non-informative censoring).

* **Collett, D. (2023). Modelling Survival Data in Medical Research** (4th ed.). Chapman and Hall/CRC. DOI: 10.1201/9781003282525
    * Covers both the Kaplan-Meier and Cox models with clarity and depth, using real-life medical examples and datasets. Suitable for graduate students and professionals.

* **Kleinbaum, D. G., & Klein, M. (2012). Survival Analysis: A Self-Learning Text** (3rd ed.). Springer. DOI: 10.1007/978-1-4419-6646-9
    * A reader-friendly introduction to survival models including Cox regression. Features intuitive explanations and worked examples using software like R and Stata.

* **Tibshirani, R. (1997). The Lasso Method for Variable Selection in the Cox Model.** Statistics in Medicine.
    * Introduces the LASSO method for automatic variable selection in Cox regression by penalizing less important coefficients.

* **Lin, Y., & Zhang, H. H. (2006). Component selection and smoothing in smoothing spline analysis of variance models.** Annals of Statistics, 34(5), 2272–2297. DOI: 10.1214/009053606000000604
    * This methodological paper introduces the COSSO framework, a unified approach for variable selection and function estimation in nonparametric regression models. Though not specific to survival analysis, later extensions have adapted COSSO to Cox models, enabling flexible modeling of non-linear covariate effects with automatic selection.

* **Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1995). A Simulation Study of the Number of Events per Variable in Logistic Regression Analysis.** Journal of Clinical Epidemiology.
    * Classic study that established the practical rule of 10 events per variable (EPV) to ensure estimate stability.

* **Vittinghoff, E., & McCulloch, C. E. (2007). Relaxing the Rule of Ten Events per Variable in Logistic and Cox Regression.** American Journal of Epidemiology, 165(6), 710–718. DOI: 10.1093/aje/kwk052

    * This study uses simulation methods to show that Cox regression models with as few as 5–9 events per variable (EPV) can produce reliable estimates, especially when effects are strong or model complexity is controlled with techniques like regularization or bootstrapping.

* **Harrell, F. E. Jr. (2015). Regression Modeling Strategies.**
    * Covers variable selection, handling multicollinearity, use of splines for nonlinear effects, and strategies to deal with censoring.

* **Steyerberg, E. W. (2019). Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating** (2nd ed.). Springer. DOI: 10.1007/978-3-030-16399-0  
    * Comprehensive guide to model development and validation, including internal and external validation techniques, calibration methods such as calibration belts, and performance metrics like the C-index and Brier score. Widely used in clinical research and survival modeling.

* **Gönen, M., & Heller, G. (2005). Concordance probability and discriminatory power in proportional hazards regression.** *Biometrika*, 92(4), 965–970. DOI: 10.1093/biomet/92.4.965  
    * Foundational paper on the mathematical definition and interpretation of the concordance index (C-index) in proportional hazards models. Provides a solid theoretical basis for discrimination assessment in survival analysis.

* **Royston, P., & Altman, D. G. (2010). Visualizing and assessing discrimination in the logistic regression model using the C-statistic.** *Statistics in Medicine*, 29(24), 2506–2516. DOI: 10.1002/sim.3994  
    * Discusses visualization and interpretation of the C-statistic, offering insights into model discrimination. While focused on logistic models, its principles are often applied to C-index interpretation in survival analysis.

* **Heagerty, P. J., & Zheng, Y. (2005). Survival model predictive accuracy and ROC curves.** *Biometrics*, 61(1), 92–105. DOI: 10.1111/j.0006-341X.2005.030814.x  
     * Introduces the concept of time-dependent ROC curves and AUC for censored survival data. Offers formal methods to evaluate predictive accuracy of Cox models at specific time points. A key reference for dynamic discrimination measures in survival analysis.


---
