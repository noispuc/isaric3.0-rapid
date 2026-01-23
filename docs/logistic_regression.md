# `rapid_regression.RAPID_LogisticRegression`

`class logistic_regression.RAPID_LogisticRegression(data, outcome_str, predictors_list, regression_type='Multi', classification_threshold=0.5)` [[source]](__https://github.com/your-repo/logistic_regression.py__)

The `RAPID_LogisticRegression` class is a specialized pipeline designed for **Logistic Regression Analysis** within epidemiological and clinical research contexts. It facilitates the end-to-end workflow from raw data cleaning to model fitting, diagnostic testing, classification performance evaluation, and the generation of publication-ready reports and visualizations.

This class inherits from `RAPID_BaseRegression` and implements a modular structure that ensures consistency across different analytical tasks.

---

### Parameters

* **data** (*pd.DataFrame*):  
  The input dataset containing the variables for analysis.

* **outcome_str** (*str*):  
  The name of the column representing the binary outcome variable (dependent variable). Should contain 0/1 or boolean values.

* **predictors_list** (*list of str*):  
  A list of feature names to be used as predictors (independent variables) in the regression model.

* **regression_type** (*str, default='Multi'*):  
  Specifies whether to perform univariate (`'Uni'`) or multivariate (`'Multi'`) regression analysis.

* **classification_threshold** (*float, default=0.5*):  
  Probability threshold for converting predicted probabilities to class labels. Values ≥ threshold are classified as 1, otherwise 0.

---

### Methods

#### `preprocess_data()`

Triggers the internal cleaning and preprocessing workflow, including handling missing values and design matrix generation with optional intercept term.

* **Returns:**  
  * None (modifies internal attributes `self.X`, `self.y`, `self.XList`)

---

#### `fit(labels=None, cross_val=True, n_splits=5)`

Executes the full modeling sequence. If the data has not been preprocessed, it runs `preprocess_data` automatically. Performs model fitting, assumption testing, performance metric calculation, and optional cross-validation.

* **Parameters:**
  * **labels** (*dict, optional*): A dictionary mapping internal column names to "pretty" labels for reporting (e.g., `{'age': 'Age (years)', 'smoking': 'Smoking Status'}`).
  * **cross_val** (*bool, default=True*): Whether to perform k-fold cross-validation to assess model generalizability.
  * **n_splits** (*int, default=5*): Number of folds for cross-validation.

---

#### `summary(assumptions=False, performance=False, cross_val=False, plots=None, vif_threshold=5.0)`

Reports the logistic regression findings and triggers visualization and diagnostic tools.

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
  * **cross_val** (*bool, default=False*): If `True`, displays cross-validation results including accuracy scores across folds.
  * **plots** (*list of str, optional*): Types of plots to generate. Supported options:
    * `'forest_plot'` - Odds ratios with confidence intervals (log scale)
    * `'roc_curve'` - ROC curve with AUC score
    * `'confusion_matrix'` - Confusion matrix heatmap
  * **vif_threshold** (*float, default=5.0*): Threshold for flagging problematic multicollinearity (VIF > threshold).

---

### Parent Class: `RAPID_BaseRegression`

The `RAPID_LogisticRegression` class inherits core functionality from `RAPID_BaseRegression`, which provides:

#### Common Attributes
* **data** (*pd.DataFrame*): Copy of the input dataset
* **outcome_str** (*str*): Outcome variable name
* **predictors_list** (*list*): List of predictor variable names
* **regression_type** (*str*): Type of regression ('Uni' or 'Multi')
* **X** (*pd.DataFrame*): Design matrix with predictors
* **y** (*pd.Series*): Outcome vector
* **model** (*statsmodels GLM result*): Fitted regression model
* **summary_df** (*pd.DataFrame*): Results summary with odds ratios and CIs

#### Shared Methods
* **preprocess_data()**: Data cleaning and preprocessing
* **fit()**: Model fitting with evaluation
* **summary()**: Abstract method for reporting (implemented by subclasses)

#### Assumption Testing (Inherited)
* **_setup_assumption_tester()**: Initializes the `ModelAssumptionTester`
* **_evaluate_vif()**: Calculates Variance Inflation Factors
* **_report_vif()**: Reports VIF results with threshold-based warnings

---

### Internal Workflow (Methods)

As a subclass of `RAPID_BaseRegression`, this class implements several private methods:

| Method | Description |
| --- | --- |
| `_data_cleaning` | Removes rows with missing values in outcome and predictor columns. |
| `_preprocessing` | Uses `RapidPreprocessor` to create design matrix with intercept. |
| `_modeling` | Fits a statsmodels GLM with Binomial family (logistic link function). |
| `_model_evaluation` | Executes assumption tests, performance metrics, and cross-validation. |
| `_test_assumptions` | Runs VIF and influential outlier diagnostics. |
| `_test_performance_metrics` | Calculates accuracy, precision, recall, F1, log loss, and AUC. |
| `_test_cross_validation` | Performs k-fold cross-validation using sklearn. |
| `_build_result_summary_df` | Constructs summary DataFrame with odds ratios and confidence intervals. |
| `_rename_cols_by_regression_type` | Adds (uni) or (multi) suffixes to column names. |

---

### Diagnostic Tests

#### 1. Multicollinearity (VIF)
Detects correlation among predictors.
- **Thresholds**:
  - VIF < 5: Low multicollinearity (acceptable)
  - VIF 5-10: Moderate multicollinearity (caution)
  - VIF > 10: High multicollinearity (problematic)

#### 2. Influential Outliers (Cook's Distance)
Identifies observations with undue influence on the regression.
- **Threshold**: 4/n (where n = sample size)
- **Interpretation**: Points above threshold may be influential outliers

---

### Classification Performance Metrics

| Metric | Description | Interpretation |
| --- | --- | --- |
| **Accuracy** | Proportion of correct predictions | Overall correctness (0-1, higher is better) |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are correct (0-1, higher is better) |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were caught (0-1, higher is better) |
| **F1 Score** | Harmonic mean of precision and recall | Balanced metric (0-1, higher is better) |
| **Log Loss** | Cross-entropy loss | Penalizes confident wrong predictions (lower is better) |
| **ROC AUC** | Area under ROC curve | Discriminatory ability (0-1, higher is better) |

---

### Examples

#### Example 1: Basic Multivariate Logistic Regression

```python
from logistic_regression import RAPID_LogisticRegression

# 1. Initialize the pipeline
model = RAPID_LogisticRegression(
    data=clinical_df,
    outcome_str='hospital_death',
    predictors_list=['age', 'sex', 'bmi', 'smoking_status', 'diabetes'],
    regression_type='Multi'
)

# 2. Fit the model
model.fit(cross_val=True, n_splits=5)

# 3. View complete summary with all diagnostics
model.summary(
    assumptions=True,
    performance=True,
    cross_val=True,
    plots=['forest_plot', 'roc_curve', 'confusion_matrix'],
    vif_threshold=5.0
)
```

#### Example 2: Univariate Analysis with Custom Labels

```python
# 1. Initialize for single predictor
model = RAPID_LogisticRegression(
    data=clinical_df,
    outcome_str='icu_admission',
    predictors_list=['sepsis'],
    regression_type='Uni'
)

# 2. Fit with custom labels
model.fit(
    labels={'sepsis': 'Sepsis Diagnosis'},
    cross_val=False
)

# 3. Simple performance summary
model.summary(performance=True, plots=['forest_plot'])
```

#### Example 3: ROC Analysis and Model Evaluation

```python
# 1. Initialize the pipeline
model = RAPID_LogisticRegression(
    data=clinical_df,
    outcome_str='cardiovascular_event',
    predictors_list=['age', 'bmi', 'systolic_bp', 'diabetes', 'smoking_status'],
    regression_type='Multi'
)

# 2. Fit the model
model.fit(cross_val=False)

# 3. Check classification performance and ROC curve
model.summary(
    performance=True,
    plots=['roc_curve', 'confusion_matrix']
)
```

#### Example 4: Publication-Ready Output

```python
# 1. Initialize with all relevant predictors
predictors = ['age', 'sex', 'bmi', 'smoking_status', 'diabetes', 'hypertension']

model = RAPID_LogisticRegression(
    data=clinical_df,
    outcome_str='mortality_30day',
    predictors_list=predictors,
    regression_type='Multi'
)

# 2. Fit with descriptive labels
labels = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'bmi': 'Body Mass Index (kg/m²)',
    'smoking_status': 'Current Smoker',
    'diabetes': 'Type 2 Diabetes',
    'hypertension': 'Hypertension'
}

model.fit(labels=labels, cross_val=True, n_splits=10)

# 3. Generate comprehensive report
model.summary(
    assumptions=True,
    performance=True,
    cross_val=True,
    plots=['forest_plot', 'roc_curve'],
    vif_threshold=5.0
)

# Access the results DataFrame for export
results_table = model.summary_df
results_table.to_csv('logistic_regression_results.csv', index=False)
```

---

### Accessing Results Programmatically

After fitting, you can access various attributes:

```python
# Summary table with odds ratios
print(model.summary_df)

# Model object (statsmodels GLM)
print(model.model.summary())

# Classification metrics
print(f"Accuracy: {model.accuracy:.3f}")
print(f"Precision: {model.precision:.3f}")
print(f"Recall: {model.recall:.3f}")
print(f"F1 Score: {model.f1:.3f}")
print(f"ROC AUC: {model.auc:.3f}")

# Assumption test results
print(model.vif_results)
print(f"Influential points: {model.influential_points}")

# Cross-validation results
if hasattr(model, 'cross_val_scores'):
    print(f"CV Accuracy: {model.cross_val_scores}")
    print(f"Mean CV Accuracy: {np.mean(model.cross_val_scores):.3f}")
```

---

### Notes

> [!IMPORTANT]
> This pipeline uses **statsmodels GLM with Binomial family** (logit link function) for logistic regression. This provides a unified interface with other regression types but uses **z-statistics** instead of t-statistics. For large samples, results are equivalent to maximum likelihood estimation.

> [!WARNING]
> The pipeline automatically drops rows with `NaN` values in the `outcome_str` or any variable in `predictors_list` during the cleaning phase. Ensure missing data is appropriately handled before analysis or use imputation methods if needed.

> [!TIP]
> For highly correlated predictors (VIF > 10), consider:
> - Removing redundant variables
> - Using regularized logistic regression (L1/L2 penalties)
> - Combining correlated variables into composite scores

---

### Statistical Notes

**Model Specification:**
The fitted model takes the form:

```
log(p / (1-p)) = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ
```

Where:
- p = P(Y=1|X) = probability of the outcome
- p/(1-p) = odds of the outcome
- β₀ = intercept (automatically included)
- βᵢ = log odds ratio for predictor i
- exp(βᵢ) = odds ratio for predictor i

**Interpreting Odds Ratios:**
- **OR = 1**: No association (predictor has no effect)
- **OR > 1**: Increased odds (e.g., OR = 2.0 means 2× higher odds)
- **OR < 1**: Decreased odds (e.g., OR = 0.5 means 50% lower odds)

**Assumptions:**
1. **Binary outcome**: Y must be 0 or 1
2. **Independence**: Observations are independent
3. **Linearity**: Linear relationship between predictors and log odds
4. **No perfect multicollinearity**: Predictors are not perfectly correlated
5. **Large sample size**: Generally need 10-15 events per predictor

The `summary()` method with `assumptions=True` tests multicollinearity and identifies influential outliers systematically.