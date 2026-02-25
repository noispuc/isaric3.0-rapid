# `rapid_survival.RAPID_survival`

`class survival_pipeline.RAPID_survival_cox(data, duration_col, event_col, predictors)` [[source]](https://www.google.com/search?q=https://github.com/your-repo/survival_pipeline.py)

The `RAPID_survival_cox` class is a specialized pipeline designed for **Survival Analysis** within epidemiological contexts. It facilitates the end-to-end workflow from raw data cleaning to the estimation of Cox Proportional Hazards models and the generation of clinical reports.

This class inherits from `RAPID_Pipeline` and implements a modular structure that ensures consistency across different analytical tasks.

### Parameters

* **data** (*pd.DataFrame*):
The input dataset containing the variables for analysis.
* **duration_col** (*str*):
The name of the column representing time-to-event (e.g., days until discharge or death).
* **event_col** (*str*):
The name of the binary column where `1` indicates the event occurred and `0` indicates censoring.
* **predictors** (*list of str*):
A list of feature names to be used as covariates in the survival model.

---

### Methods

#### `fit(formula=None, labels=None, penalizer=0.1)`

Executes the full modeling sequence. If the data has not been preprocessed, it runs `preprocess_data` automatically.

* **Parameters:**
* **formula** (*str, optional*): A dictionary mapping internal column names to "pretty" labels for reporting.
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

> [!IMPORTANT]
> In this version, survival analysis is conducted solely using the listwise deletion method to address missing data.

---