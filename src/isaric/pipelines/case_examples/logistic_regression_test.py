"""
Testing script for RAPID_LogisticRegression module.
Tests all functionality including fit, assumptions, performance metrics, plots,
cross-validation, formula interface, and all supported link functions.
"""

import numpy as np
import pandas as pd
import warnings
import traceback
warnings.filterwarnings('ignore')

from isaric.pipelines.logistic_regression import RAPID_LogisticRegression

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

def generate_test_data(n_samples=300, random_state=42):
    """
    Generate synthetic binary outcome data for logistic regression testing.
    Includes both continuous and categorical predictors.
    """
    np.random.seed(random_state)

    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)

    sex = np.random.choice(['Male', 'Female'], n_samples)
    smoking = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

    # Log-odds linear combination
    log_odds = (
        -3.0 +
        0.04 * age +
        0.08 * bmi -
        0.01 * blood_pressure +
        0.5 * (sex == 'Male').astype(int) +
        0.8 * (smoking == 'Yes').astype(int)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, prob, n_samples)

    df = pd.DataFrame({
        'outcome': y,
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'sex': sex,
        'smoking_status': smoking
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
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Model fitted successfully")
    print(f"  - Number of observations: {len(df)}")
    print(f"  - Number of predictors: {len(predictors)}")
    print(f"  - Model type: {model.regression_type}")

    return model


def test_with_labels():
    """Test model fitting with custom labels."""
    print("\n" + "=" * 80)
    print("TEST 2: Model Fitting with Custom Labels")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex',
        'smoking_status': 'Smoking Status'
    }

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
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
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- ALL ASSUMPTION TEST RESULTS ---\n")
    model.summary(assumptions='all')

    print("\n--- SELECTED ASSUMPTION METRICS ---\n")
    model.summary(assumptions=['Events Per Variable (EPV)'])

    print("\n✓ All assumption tests completed")

    return model


def test_assumptions_vif():
    """Test that VIF is shown when explicitly requested."""
    print("\n" + "=" * 80)
    print("TEST 4: Assumptions — VIF Selection")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
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
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
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
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- ALL PERFORMANCE METRICS ---\n")
    model.summary(performance='all')

    print("\n--- SELECTED PERFORMANCE METRICS ---\n")
    model.summary(performance=['Accuracy', 'AUC-ROC', 'F1 Score'])

    print("\n✓ Performance metrics calculated")

    return model


def test_performance_confusion_matrix():
    """Test that confusion matrix is shown when explicitly requested."""
    print("\n" + "=" * 80)
    print("TEST 7: Performance — Confusion Matrix Selection")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- CONFUSION MATRIX ONLY ---\n")
    model.summary(performance=['Confusion Matrix'])

    print("\n--- ACCURACY + CONFUSION MATRIX ---\n")
    model.summary(performance=['Accuracy', 'Confusion Matrix'])

    print("\n✓ Confusion matrix selection test completed")

    return model


def test_cross_validation():
    """Test cross-validation — 'all' and selected metrics."""
    print("\n" + "=" * 80)
    print("TEST 8: Cross-Validation")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=True, n_splits=5)

    print("\n--- ALL CROSS-VALIDATION RESULTS ---\n")
    model.summary(cross_val='all')

    print("\n--- SELECTED CV METRICS ---\n")
    model.summary(cross_val=['Mean Accuracy', 'Standard Deviation'])

    print("\n✓ Cross-validation completed")

    return model


def test_plots():
    """Test all plotting functions."""
    print("\n" + "=" * 80)
    print("TEST 9: Plotting Functions")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex']

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex'
    }

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(labels=labels, cross_val=False)

    print("\n--- GENERATING PLOTS ---\n")

    plots_to_generate = ['forest_plot', 'roc_curve', 'confusion_matrix']

    for plot_name in plots_to_generate:
        print(f"Generating {plot_name}...")
        model.summary(plots=[plot_name])

    print("\n✓ All plots generated successfully")

    return model


def test_univariate():
    """Test univariate regression."""
    print("\n" + "=" * 80)
    print("TEST 10: Univariate Regression")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['bmi'],
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
    print("TEST 11: Multicollinearity Detection (VIF)")
    print("=" * 80)

    df = generate_test_data(n_samples=400)

    df['bmi_squared'] = df['bmi'] ** 2
    df['age_bmi_interaction'] = df['age'] * df['bmi']

    predictors = ['age', 'bmi', 'bmi_squared', 'age_bmi_interaction', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n--- VIF TEST (with threshold = 5.0) ---\n")
    model.summary(assumptions=['VIF'], vif_threshold=5.0)

    print("\n--- VIF TEST (with threshold = 10.0) ---\n")
    model.summary(assumptions=['VIF'], vif_threshold=10.0)

    print("\n✓ Multicollinearity test completed")

    return model


def test_classification_threshold():
    """Test custom classification threshold."""
    print("\n" + "=" * 80)
    print("TEST 12: Custom Classification Threshold")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n  Testing threshold = {threshold}")
        model = RAPID_LogisticRegression(
            data=df,
            yvar='outcome',
            predictors=predictors,
            regression_type='Multi',
            classification_threshold=threshold
        )
        model.fit(cross_val=False)
        print(f"  Accuracy at threshold {threshold}: {model.accuracy:.4f}")

    print("\n✓ Classification threshold tests completed")

    return model


def test_complete_pipeline():
    """Test complete analysis pipeline."""
    print("\n" + "=" * 80)
    print("TEST 13: Complete Analysis Pipeline")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex',
        'smoking_status': 'Smoking Status'
    }

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=predictors,
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
        plots=['forest_plot', 'roc_curve', 'confusion_matrix'],
        vif_threshold=5.0
    )

    print("\n✓ Complete pipeline executed successfully")

    return model


# ============================================================================
# FORMULA INTERFACE
# ============================================================================

def test_formula_interface():
    """Test that the formula interface works as an alternative to yvar + predictors."""
    print("\n" + "=" * 80)
    print("TEST 14: Formula Interface")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
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
    """Test formula interface with categorical predictors via C() wrapper."""
    print("\n" + "=" * 80)
    print("TEST 15: Formula Interface with Categorical Predictors")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        formula='outcome ~ age + bmi + C(sex) + C(smoking_status)',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Formula with categorical predictors fitted successfully")
    print("\nSummary DataFrame:")
    print(model.summary_df)

    return model


def test_formula_with_labels():
    """Test that labels are applied correctly when using the formula interface."""
    print("\n" + "=" * 80)
    print("TEST 16: Formula Interface with Labels")
    print("=" * 80)

    df = generate_test_data()

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
    }

    model = RAPID_LogisticRegression(
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
    print("TEST 17: Formula Interface with Cross-Validation")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
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
# LINK FUNCTION TESTS
# ============================================================================

def test_binomial_logit_link():
    """Test Binomial family with logit link (canonical, default)."""
    print("\n" + "=" * 80)
    print("TEST 18: Binomial — Logit Link (Canonical, Default)")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['age', 'bmi', 'blood_pressure'],
        family='binomial',
        link='logit',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Binomial + logit fitted successfully")
    model.summary(performance='all')

    return model


def test_binomial_probit_link():
    """Test Binomial family with probit link."""
    print("\n" + "=" * 80)
    print("TEST 19: Binomial — Probit Link")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['age', 'bmi', 'blood_pressure'],
        family='binomial',
        link='probit',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Binomial + probit fitted successfully")
    model.summary(performance='all')

    return model


def test_binomial_cloglog_link():
    """Test Binomial family with complementary log-log link."""
    print("\n" + "=" * 80)
    print("TEST 20: Binomial — Complementary Log-Log Link")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['age', 'bmi', 'blood_pressure'],
        family='binomial',
        link='cloglog',
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("\n✓ Binomial + cloglog fitted successfully")
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

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['age', 'bmi', 'blood_pressure'],
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

    model = RAPID_LogisticRegression(
        data=df,
        yvar='outcome',
        predictors=['age', 'bmi', 'blood_pressure'],
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
    print("█" + " " * 19 + "RAPID LOGISTIC REGRESSION TEST SUITE" + " " * 23 + "█")
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
        test_performance_confusion_matrix,
        test_cross_validation,
        test_plots,
        test_univariate,
        test_multicollinearity,
        test_classification_threshold,
        test_complete_pipeline,
        # Formula interface
        test_formula_interface,
        test_formula_with_categorical,
        test_formula_with_labels,
        test_formula_cross_validation,
        # Link function combinations
        test_binomial_logit_link,
        test_binomial_probit_link,
        test_binomial_cloglog_link,
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
            traceback.print_exc()

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