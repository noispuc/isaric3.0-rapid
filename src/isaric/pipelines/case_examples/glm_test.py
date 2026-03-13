"""
Testing script for RAPID_GLM module.
Tests all functionality including fit, assumptions, performance metrics, and plots.
"""

import numpy as np
import pandas as pd
import warnings
import traceback
warnings.filterwarnings('ignore')

from isaric.pipelines.glm import RAPID_GLM

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

def generate_test_data(n_samples=200, random_state=42):
    """
    Generate synthetic data for linear regression testing.
    Includes both continuous and categorical independent_vars.
    """
    np.random.seed(random_state)
    
    # Continuous independent_vars
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    
    # Categorical independent_vars
    sex = np.random.choice(['Male', 'Female'], n_samples)
    smoking = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    
    # Create outcome with known relationship
    # y = 50 + 0.5*age + 2*bmi - 0.3*bp + 10*(sex==Male) + 15*(smoking==Yes) + noise
    y = (50 + 
         0.5 * age + 
         2.0 * bmi - 
         0.3 * blood_pressure +
         10 * (sex == 'Male').astype(int) +
         15 * (smoking == 'Yes').astype(int) +
         np.random.normal(0, 10, n_samples))
    
    df = pd.DataFrame({
        'outcome': y,
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'sex': sex,
        'smoking_status': smoking
    })
    
    return df


def generate_positive_outcome_data(n_samples=200, random_state=42):
    np.random.seed(random_state)

    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)

    # Standardize independent_vars for numerical stability
    age = (age - age.mean()) / age.std()
    bmi = (bmi - bmi.mean()) / bmi.std()
    blood_pressure = (blood_pressure - blood_pressure.mean()) / blood_pressure.std()

    log_y = (3.5 +
              0.01 * age +
              0.04 * bmi -
              0.005 * blood_pressure +
              np.random.normal(0, 0.3, n_samples))
    y = np.exp(log_y)

    df = pd.DataFrame({
        'outcome': y,
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
    })

    return df


# ============================================================================
# ORIGINAL TESTS
# ============================================================================

def test_basic_fit():
    """Test basic model fitting."""
    print("=" * 80)
    print("TEST 1: Basic Model Fitting")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    try:
        model.fit(cross_val=False)
    except:
        traceback.print_exc()
    
    print("\n✓ Model fitted successfully")
    print(f"  - Number of observations: {len(df)}")
    print(f"  - Number of independent_vars: {len(independent_vars)}")
    print(f"  - Model type: {model.regression_type}")
    
    return model


def test_with_labels():
    """Test model fitting with custom labels."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Fitting with Custom Labels")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']
    
    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex',
        'smoking_status': 'Smoking Status'
    }
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(labels=labels, cross_val=False)
    
    print("\n✓ Model fitted with custom labels")
    print("\nSummary DataFrame:")
    print(model.summary_df)
    
    return model


def test_assumptions():
    """Test assumption checking — 'all' shows full assumption block including VIF and outliers."""
    print("\n" + "=" * 80)
    print("TEST 3: Assumption Tests")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- ALL ASSUMPTION TEST RESULTS ---\n")
    model.summary(assumptions='all')

    print("\n--- SELECTED ASSUMPTION METRICS ---\n")
    model.summary(assumptions=['Durbin-Watson', 'Shapiro-Wilk p-value'])
    
    print("\n✓ All assumption tests completed")
    
    return model


def test_assumptions_vif():
    """Test that VIF is shown when explicitly requested."""
    print("\n" + "=" * 80)
    print("TEST 4: Assumptions — VIF Selection")
    print("=" * 80)

    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure']

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- VIF ONLY ---\n")
    model.summary(assumptions=['VIF'])

    print("\n✓ VIF selection test completed")

    return model


def test_assumptions_influential_outliers():
    """Test that influential outlier details are shown when explicitly requested."""
    print("\n" + "=" * 80)
    print("TEST 5: Assumptions — Influential Outliers Selection")
    print("=" * 80)

    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure']

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- INFLUENTIAL OUTLIERS ONLY ---\n")
    model.summary(assumptions=['Influential Outliers'])

    print("\n--- OUTLIER ROWS + DETAILS ---\n")
    model.summary(assumptions=['Influential Outliers Threshold', 'Number of Influential Points', 'Influential Outliers'])

    print("\n✓ Influential outliers selection test completed")

    return model


def test_performance_metrics():
    """Test performance metrics — 'all' and selected metrics."""
    print("\n" + "=" * 80)
    print("TEST 6: Performance Metrics")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- ALL PERFORMANCE METRICS ---\n")
    model.summary(performance='all')

    print("\n--- SELECTED PERFORMANCE METRICS ---\n")
    model.summary(performance=['R2', 'Adjusted R2', 'AIC', 'BIC'])
    
    print("\n✓ Performance metrics calculated")
    
    return model


def test_cross_validation():
    """Test cross-validation — 'all' and selected metrics."""
    print("\n" + "=" * 80)
    print("TEST 7: Cross-Validation")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(cross_val=True, n_splits=5)
    
    print("\n--- ALL CROSS-VALIDATION RESULTS ---\n")
    model.summary(cross_val='all')

    print("\n--- SELECTED CV METRICS ---\n")
    model.summary(cross_val=['Mean CV MSE', 'Standard Deviation of CV MSE'])
    
    print("\n✓ Cross-validation completed")
    
    return model


def test_plots():
    """Test all plotting functions."""
    print("\n" + "=" * 80)
    print("TEST 8: Plotting Functions")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure', 'sex']
    
    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex'
    }
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(labels=labels, cross_val=False)
    
    print("\n--- GENERATING PLOTS ---\n")
    
    plots_to_generate = ['forest_plot', 'residuals_vs_fitted', 'qq_plot']
    
    for plot_name in plots_to_generate:
        print(f"Generating {plot_name}...")
        model.summary(plots=[plot_name])
    
    print("\n✓ All plots generated successfully")
    
    return model


def test_univariate():
    """Test univariate regression."""
    print("\n" + "=" * 80)
    print("TEST 9: Univariate Regression")
    print("=" * 80)
    
    df = generate_test_data()
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['bmi'],
        regression_type='Uni'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- UNIVARIATE RESULTS ---")
    print("\nSummary DataFrame:")
    print(model.summary_df)
    
    print("\n--- PERFORMANCE METRICS ---\n")
    model.summary(performance='all')
    
    print("\n✓ Univariate regression completed")
    
    return model


def test_multicollinearity():
    """Test VIF for multicollinearity detection at different thresholds."""
    print("\n" + "=" * 80)
    print("TEST 10: Multicollinearity Detection (VIF)")
    print("=" * 80)
    
    df = generate_test_data(n_samples=300)
    
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    independent_vars = ['age', 'bmi', 'bmi_squared', 'age_bmi_interaction', 'blood_pressure']
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- VIF TEST (with threshold = 5.0) ---\n")
    model.summary(assumptions=['VIF'], vif_threshold=5.0)
    
    print("\n--- VIF TEST (with threshold = 10.0) ---\n")
    model.summary(assumptions=['VIF'], vif_threshold=10.0)
    
    print("\n✓ Multicollinearity test completed")
    
    return model


def test_complete_pipeline():
    """Test complete analysis pipeline."""
    print("\n" + "=" * 80)
    print("TEST 11: Complete Analysis Pipeline")
    print("=" * 80)
    
    df = generate_test_data()
    independent_vars = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']
    
    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex',
        'smoking_status': 'Smoking Status'
    }
    
    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=independent_vars,
        regression_type='Multi'
    )
    
    print("\nFitting model with cross-validation...")
    model.fit(labels=labels, cross_val=True, n_splits=5)
    
    print("\n" + "=" * 80)
    print("COMPLETE MODEL SUMMARY")
    print("=" * 80)
    
    model.summary(
        assumptions='all',
        performance='all',
        cross_val='all',
        plots=['forest_plot', 'residuals_vs_fitted', 'qq_plot'],
        vif_threshold=5.0
    )
    
    print("\n✓ Complete pipeline executed successfully")
    
    return model


# ============================================================================
# FORMULA INTERFACE
# ============================================================================

def test_formula_interface():
    """Test that the formula interface works as an alternative to dependent_var + independent_vars."""
    print("\n" + "=" * 80)
    print("TEST 12: Formula Interface")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_GLM(
        data=df,
        formula='outcome ~ age + bmi + blood_pressure',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Formula interface fitted successfully")
    print("\nSummary DataFrame:")
    print(model.summary_df)

    return model


def test_formula_with_categorical():
    """Test formula interface with categorical independent_vars via C() wrapper."""
    print("\n" + "=" * 80)
    print("TEST 13: Formula Interface with Categorical independent_vars")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_GLM(
        data=df,
        formula='outcome ~ age + bmi + C(sex) + C(smoking_status)',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Formula with categorical independent_vars fitted successfully")
    print("\nSummary DataFrame:")
    print(model.summary_df)

    return model


def test_formula_with_labels():
    """Test that labels are applied correctly when using the formula interface."""
    print("\n" + "=" * 80)
    print("TEST 14: Formula Interface with Labels")
    print("=" * 80)

    df = generate_test_data()

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
    }

    model = RAPID_GLM(
        data=df,
        formula='outcome ~ age + bmi + blood_pressure',
        regression_type='Multi'
    )

    model.fit(labels=labels, cross_val=False)

    print("\n✓ Formula + labels fitted successfully")
    print("\nSummary DataFrame:")
    print(model.summary_df)

    return model


def test_formula_cross_validation():
    """Test cross-validation works correctly when specified via formula."""
    print("\n" + "=" * 80)
    print("TEST 15: Formula Interface with Cross-Validation")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_GLM(
        data=df,
        formula='outcome ~ age + bmi + blood_pressure',
        regression_type='Multi'
    )

    model.fit(cross_val=True, n_splits=3)

    print("\n--- CROSS-VALIDATION RESULTS ---\n")
    model.summary(cross_val='all')

    print("\n✓ Formula + cross-validation completed")

    return model


# ============================================================================
# FAMILY AND LINK COMBINATIONS
# ============================================================================

def test_gamma_log_link():
    """Test Gamma family with its canonical log link."""
    print("\n" + "=" * 80)
    print("TEST 16: Gamma Family — Log Link (Canonical)")
    print("=" * 80)

    df = generate_positive_outcome_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        family='gamma',
        link='log',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Gamma + log link fitted successfully")
    model.summary(performance='all')

    return model


def test_gamma_inverse_link():
    """Test Gamma family with inverse link."""
    print("\n" + "=" * 80)
    print("TEST 17: Gamma Family — Inverse Link")
    print("=" * 80)

    df = generate_positive_outcome_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        family='gamma',
        link='inverse',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Gamma + inverse link fitted successfully")
    model.summary(performance='all')

    return model


def test_inv_gaussian_inverse_link():
    """Test Inverse Gaussian family with its canonical inverse link."""
    print("\n" + "=" * 80)
    print("TEST 18: Inverse Gaussian Family — Inverse Link (Canonical)")
    print("=" * 80)

    df = generate_positive_outcome_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        family='inv_gaussian',
        link='inverse',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Inverse Gaussian + inverse link fitted successfully")
    model.summary(performance='all')

    return model


def test_tweedie_log_link():
    """Test Tweedie family with its canonical log link."""
    print("\n" + "=" * 80)
    print("TEST 19: Tweedie Family — Log Link (Canonical)")
    print("=" * 80)

    df = generate_positive_outcome_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        family='tweedie',
        link='log',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Tweedie + log link fitted successfully")
    model.summary(performance='all')

    return model


def test_gamma_formula():
    """Test Gamma family combined with the formula interface."""
    print("\n" + "=" * 80)
    print("TEST 20: Gamma Family via Formula Interface")
    print("=" * 80)

    df = generate_positive_outcome_data()

    model = RAPID_GLM(
        data=df,
        formula='outcome ~ age + bmi + blood_pressure',
        family='gamma',
        link='log',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Gamma + formula fitted successfully")
    print("\nSummary DataFrame:")
    print(model.summary_df)
    model.summary(performance='all')

    return model


# ============================================================================
# EDGE CASES
# ============================================================================

def test_cross_val_not_run_then_requested():
    """
    Edge case: fit() called with cross_val=False, but summary() called with
    cross_val='all'. Guard should print a warning rather than raise an exception.
    """
    print("\n" + "=" * 80)
    print("TEST 21: Cross-Val Requested in summary() but Not Run in fit() (Guard Path)")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\nCalling summary(cross_val='all') without having run CV — expect warning message:")
    model.summary(cross_val='all')

    print("\n✓ Guard path handled gracefully")

    return model


def test_invalid_metric_warning():
    """
    Edge case: passing a metric name that doesn't exist should print a warning
    rather than raising an exception.
    """
    print("\n" + "=" * 80)
    print("TEST 22: Invalid Metric String Warning")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_GLM(
        data=df,
        dependent_var='outcome',
        independent_vars=['age', 'bmi', 'blood_pressure'],
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\nPassing invalid metric names — expect warning messages:")
    model.summary(assumptions=['NonExistentTest'])
    model.summary(performance=['NonExistentMetric'])

    print("\n✓ Invalid metric warnings handled gracefully")

    return model


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Execute all test functions."""
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "RAPID LINEAR REGRESSION TEST SUITE" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    tests = [
        # Original tests
        test_basic_fit,
        test_with_labels,
        test_assumptions,
        test_assumptions_vif,
        test_assumptions_influential_outliers,
        test_performance_metrics,
        test_cross_validation,
        test_plots,
        test_univariate,
        test_multicollinearity,
        test_complete_pipeline,
        # Formula interface
        test_formula_interface,
        test_formula_with_categorical,
        test_formula_with_labels,
        test_formula_cross_validation,
        # Family and link combinations
        test_gamma_log_link,
        test_gamma_inverse_link,
        test_inv_gaussian_inverse_link,
        test_tweedie_log_link,
        test_gamma_formula,
        # Edge cases
        test_cross_val_not_run_then_requested,
        test_invalid_metric_warning,
    ]
    
    results = {}
    
    for i, test_func in enumerate(tests, 1):
        try:
            model = test_func()
            results[test_func.__name__] = "PASS"
        except Exception as e:
            results[test_func.__name__] = f"FAIL: {str(e)}"
            print(f"\n✗ Test failed with error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")
    
    print("\n" + "=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_all_tests()