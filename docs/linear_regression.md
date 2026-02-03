# `rapid_regression.RAPID_LinearRegression`

`class linear_regression.RAPID_LinearRegression(data, outcome_str, predictors_list, regression_type='Multi')` [[source]](__https://github.com/your-repo/linear_regression.py__)

The `RAPID_LinearRegression` class is a specialized pipeline designed for **Linear Regression Analysis** within epidemiological and clinical research contexts. It facilitates the end-to-end workflow from raw data cleaning to model fitting, diagnostic testing, and the generation of publication-ready reports and visualizations.

This class inherits from `RAPID_BaseRegression` and implements a modular structure that ensures consistency across different analytical tasks.

---

### Parameters

* **data** (*pd.DataFrame*):  
  The input dataset containing the variables for analysis.

* **outcome_str** (*str*):  
  The name of the column representing the continuous outcome variable (dependent variable).

* **predictors_list** (*list of str*):  
  A list of feature names to be used as predictors (independent variables) in the regression model.

* **regression_type** (*str, default='Multi'*):  
  Specifies whether to perform univariate (`'Uni'`) or multivariate (`'Multi'`) regression analysis.

---

### Methods

#### `fit(labels=None, cross_val=True, n_splits=5)`

Executes the full modeling sequence. If the data has not been preprocessed, it runs `preprocess_data` automatically. Performs model fitting, assumption testing, performance metric calculation, and optional cross-validation.

* **Parameters:**
  * **labels** (*dict, optional*): A dictionary mapping internal column names to "pretty" labels for reporting (e.g., `{'age': 'Age (years)', 'bmi': 'Body Mass Index'}`).
  * **cross_val** (*bool, default=True*): Whether to perform k-fold cross-validation to assess model generalizability.
  * **n_splits** (*int, default=5*): Number of folds for cross-validation.

---

#### `summary(assumptions=False, performance=False, cross_val=False, plots=None, vif_threshold=5.0)`

Reports the linear regression findings and triggers visualization and diagnostic tools.

* **Parameters:**
  * **assumptions** (*bool, default=False*): If `True`, displays results of assumption tests including:
    * Independence of Errors (Durbin-Watson)
    * Normality of Errors (Shapiro-Wilk)
    * Multicollinearity (VIF)
    * Influential Outliers (Cook's Distance)
  * **performance** (*bool, default=False*): If `True`, displays performance metrics including:
    * Mean Squared Error (MSE)
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Error (MAE)
    * R² (Coefficient of Determination)
    * Adjusted R²
  * **cross_val** (*bool, default=False*): If `True`, displays cross-validation results including mean and standard deviation of CV MSE.
  * **plots** (*list of str, optional*): Types of plots to generate. Supported options:
    * `'forest_plot'` - Coefficient estimates with confidence intervals
    * `'residuals_vs_fitted'` - Diagnostic plot for homoscedasticity
    * `'qq_plot'` - Q-Q plot for normality assessment
  * **vif_threshold** (*float, default=5.0*): Threshold for flagging problematic multicollinearity (VIF > threshold).

---

### Diagnostic Tests

#### 1. Independence of Errors (Durbin-Watson)
Tests for autocorrelation in residuals.
- **Range**: 0 to 4
- **Interpretation**:
  - < 1.5: Positive autocorrelation
  - 1.5-2.5: Independent residuals (desired)
  - > 2.5: Negative autocorrelation

#### 2. Normality of Errors (Shapiro-Wilk)
Tests whether residuals follow a normal distribution.
- **Null Hypothesis**: Residuals are normally distributed
- **Interpretation**: p > 0.05 indicates normality

#### 3. Multicollinearity (VIF)
Detects correlation among predictors.
- **Thresholds**:
  - VIF < 5: Low multicollinearity (acceptable)
  - VIF 5-10: Moderate multicollinearity (caution)
  - VIF > 10: High multicollinearity (problematic)

#### 4. Influential Outliers (Cook's Distance)
Identifies observations with undue influence on the regression.
- **Threshold**: 4/n (where n = sample size)
- **Interpretation**: Points above threshold may be influential outliers

---

### Performance Metrics

| Metric | Description | Interpretation |
| --- | --- | --- |
| **MSE** | Mean Squared Error | Average squared prediction error (lower is better) |
| **RMSE** | Root Mean Squared Error | MSE in original units (lower is better) |
| **MAE** | Mean Absolute Error | Average absolute prediction error (lower is better) |
| **R²** | Coefficient of Determination | Proportion of variance explained (0-1, higher is better) |
| **Adjusted R²** | R² adjusted for number of predictors | Penalized R² for model complexity |

---

### Examples

#### Example 1: Basic Multivariate Regression

```python
from linear_regression import RAPID_LinearRegression

# 1. Initialize the pipeline
model = RAPID_LinearRegression(
    data=clinical_df,
    outcome_str='blood_pressure',
    predictors_list=['age', 'bmi', 'sex', 'smoking_status'],
    regression_type='Multi'
)

# 2. Fit the model
model.fit(cross_val=True, n_splits=5)

# 3. View complete summary with all diagnostics
model.summary(
    assumptions=True,
    performance=True,
    cross_val=True,
    plots=['forest_plot', 'residuals_vs_fitted', 'qq_plot'],
    vif_threshold=5.0
)
```

#### Example 2: Univariate Regression with Custom Labels

```python
# 1. Initialize for single predictor
model = RAPID_LinearRegression(
    data=clinical_df,
    outcome_str='cholesterol',
    predictors_list=['bmi'],
    regression_type='Uni'
)

# 2. Fit with custom labels
model.fit(
    labels={'bmi': 'Body Mass Index (kg/m²)'},
    cross_val=False
)

# 3. Simple performance summary
model.summary(performance=True)
```

#### Example 3: Diagnostic-Focused Analysis

```python
# 1. Initialize the pipeline
model = RAPID_LinearRegression(
    data=clinical_df,
    outcome_str='glucose_level',
    predictors_list=['age', 'bmi', 'waist_circumference', 'hip_circumference'],
    regression_type='Multi'
)

# 2. Fit the model
model.fit(cross_val=False)

# 3. Check for multicollinearity and other assumptions
model.summary(
    assumptions=True,
    plots=['qq_plot', 'residuals_vs_fitted'],
    vif_threshold=5.0
)
```

#### Example 4: Publication-Ready Output

```python
# 1. Initialize with all relevant predictors
predictors = ['age', 'sex', 'bmi', 'smoking_status', 'diabetes', 'hypertension']

model = RAPID_LinearRegression(
    data=clinical_df,
    outcome_str='cardiovascular_risk_score',
    predictors_list=predictors,
    regression_type='Multi'
)

# 2. Fit with descriptive labels
labels = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'bmi': 'Body Mass Index (kg/m²)',
    'smoking_status': 'Smoking Status',
    'diabetes': 'Type 2 Diabetes',
    'hypertension': 'Hypertension'
}

model.fit(labels=labels, cross_val=True, n_splits=10)

# 3. Generate comprehensive report
model.summary(
    assumptions=True,
    performance=True,
    cross_val=True,
    plots=['forest_plot'],
    vif_threshold=5.0
)

# Access the results DataFrame for export
results_table = model.summary_df
results_table.to_csv('regression_results.csv', index=False)
```

---

### Accessing Results Programmatically

After fitting, you can access various attributes:

```python
# Summary table
print(model.summary_df)

# Model object (statsmodels GLM)
print(model.model.summary())

# Performance metrics
print(f"R²: {model.r2:.3f}")
print(f"Adjusted R²: {model.adjusted_r2:.3f}")
print(f"RMSE: {model.rmse:.3f}")

# Assumption test results
print(f"Durbin-Watson: {model.dw:.3f}")
print(f"Shapiro-Wilk p-value: {model.shapiro_wilk_p_value:.4f}")
print(model.vif_results)

# Cross-validation results
if hasattr(model, 'cv_mse_scores'):
    print(f"CV MSE: {model.cv_mse_scores}")
    print(f"Mean CV MSE: {np.mean(model.cv_mse_scores):.3f}")
```

---

### Notes

> [!IMPORTANT]
> This pipeline uses **statsmodels GLM with Gaussian family** rather than OLS directly. This provides a unified interface with other regression types (e.g., logistic regression) but uses **z-statistics** instead of t-statistics. For large samples (n > 100), the difference is negligible. For small samples, p-values may be slightly less conservative than traditional OLS.

> [!WARNING]
> The pipeline automatically drops rows with `NaN` values in the `outcome_str` or any variable in `predictors_list` during the cleaning phase. Ensure missing data is appropriately handled before analysis or use imputation methods if needed.

> [!TIP]
> For highly correlated predictors (VIF > 10), consider:
> - Removing redundant variables
> - Using ridge regression for regularization
> - Applying Principal Component Analysis (PCA)
> - Combining correlated variables into composite scores

---

### Statistical Notes

**Model Specification:**
The fitted model takes the form:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε
```

Where:
- y = outcome variable
- β₀ = intercept (automatically included)
- βᵢ = coefficients for predictors
- xᵢ = predictor variables
- ε = error term (assumed ~ N(0, σ²))

**Assumptions:**
1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of errors
4. **Normality**: Errors are normally distributed
5. **No multicollinearity**: Predictors are not highly correlated

The `summary()` method with `assumptions=True` tests these assumptions systematically.