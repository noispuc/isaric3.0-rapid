# `survival_cox.RAPID_SurvivalCox`

`class survival_pipeline.RAPID_SurvivalCox(data, duration_col, event_col, predictors)` [[source]](https://github.com/noispuc/isaric3.0-rapid/blob/main/src/isaric/pipelines/)

>The Cox Proportional Hazards Model is a widely used method in survival analysis to assess the effect of predictor variables on survival time. It is particularly valuable because it does not require specifying the underlying distribution of survival times, making it a semi-parametric model.
---

The `RAPID_SurvivalCox` class is a specialized pipeline designed for **Survival Analysis** within epidemiological contexts. It facilitates the end-to-end workflow from raw data to the estimation of Cox Proportional Hazards models and the generation of clinical reports.

This class inherits from `RAPID_Pipeline` and implements a modular structure that ensures consistency across different analytical tasks.

### Parameters

* **data** (*pd.DataFrame*):
The input dataset containing the variables for analysis.
* **duration_col** (*str*):
The name of the column representing time-to-event (e.g., days until discharge or death).
* **dependent_var** (*str*):
The name of the binary column where `1` indicates the event occurred and `0` indicates censoring.
* **independent_vars** (*list of str*):
A list of feature names to be used as covariates in the survival model.

---

### Methods

#### `fit(formula=None, labels=None, penalizer=0.1)`

Executes the full modeling sequence. If the data has not been preprocessed, it runs `preprocess_data` automatically.

* **Parameters:**
* **formula** (*str, optional*):  R-style formula string from lists of column names for data transformation using formulas.
* **labels** (*dict, optional*): A dictionary mapping internal column names to "pretty" labels for reporting.
* **penalizer** (*float, default=0.1*): L2 regularization parameter to improve model stability and prevent overfitting.



#### `summary(plots=None, target_time=None)`

Reports the Cox model findings and triggers visualization tools.

* **Parameters:**
* **plots** (*list of str, optional*): Types of plots to generate. Supported: `['forest_plot', 'roc_auc']`.
* **target_time** (*float, optional*): The specific time point used for calculating ROC curves and AUC.


### Examples

Below is a standard implementation of the survival pipeline:

```python
from survival_pipeline import RAPID_survival

# 1. Initialize the pipeline
pipeline = RAPID_survival(
    data=clinical_df,
    duration_col='days_to_event',
    event_col='outcome_death',
    predictors=['age', 'sex', 'bmi', 'comorbidity_index']
)

# 2. Fit the model with specific labels for the output
pipeline.fit(
    labels={'age': 'Age (years)', 'bmi': 'Body Mass Index'},
    penalizer=0.05
)

# 3. Generate summary and Forest Plot
pipeline.summary(plots=['forest_plot', 'roc_auc'], target_time=28)

```

### Notes

> [! IMPORTANT]

> In this version, survival analysis is conducted solely using the listwise deletion method to address missing data.

> To implement the Cox Proportional we used the <code>lifelines</code> library — a specialized package for survival analysis. It provides easy-to-use tools for fitting and interpreting models like Kaplan-Meier, Cox Proportional Hazards, and more. You can learn more about it's own official documentation at:
https://lifelines.readthedocs.io
---


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
