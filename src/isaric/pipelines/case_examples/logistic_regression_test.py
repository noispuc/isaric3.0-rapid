"""
Testing script for RAPID_LogisticRegression module.
Tests fitting, assumptions, performance metrics, cross-validation, and plots.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from isaric.pipelines.logistic_regression import RAPID_LogisticRegression

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

def generate_test_data(n_samples=300, random_state=42):
    np.random.seed(random_state)

    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    bp = np.random.normal(120, 20, n_samples)

    sex = np.random.choice(['Male', 'Female'], n_samples)
    smoking = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

    logit = (
        -6
        + 0.04 * age
        + 0.08 * bmi
        + 0.02 * bp
        + 0.8 * (sex == 'Male').astype(int)
        + 1.2 * (smoking == 'Yes').astype(int)
    )

    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob)

    return pd.DataFrame({
        'outcome': y,
        'age': age,
        'bmi': bmi,
        'blood_pressure': bp,
        'sex': sex,
        'smoking_status': smoking
    })

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_basic_fit():
    print("=" * 80)
    print("TEST 1: Basic Logistic Regression Fit")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )

    model.fit(cross_val=False)

    print("✓ Model fitted successfully")
    return model


def test_assumptions():
    print("\n" + "=" * 80)
    print("TEST 2: Assumption Checks")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(cross_val=False)
    model.summary(assumptions=True, performance=False, plots=None)

    print("✓ Assumptions evaluated")
    return model


def test_performance_metrics():
    print("\n" + "=" * 80)
    print("TEST 3: Performance Metrics")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(cross_val=False)
    model.summary(assumptions=False, performance=True, plots=None)

    print("✓ Performance metrics calculated")
    return model


def test_cross_validation():
    print("\n" + "=" * 80)
    print("TEST 4: Cross-Validation")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(cross_val=True, n_splits=5)
    model.summary(cross_val=True, assumptions=False, performance=False, plots=None)

    print("✓ Cross-validation completed")
    return model


def test_plots():
    print("\n" + "=" * 80)
    print("TEST 5: Plotting")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(cross_val=False)

    # Fixed: removed duplicate 'plots =' 
    plots = ['forest_plot', 'roc_curve', 'confusion_matrix']
    model.summary(assumptions=False, performance=False, plots=plots)

    # Removed: Don't call internal methods - plots already shown above
    print("✓ All plots generated")
    return model


def test_univariate():
    print("\n" + "=" * 80)
    print("TEST 6: Univariate Logistic Regression")
    print("=" * 80)

    df = generate_test_data()

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=['bmi'],
        regression_type='Uni'
    )

    model.fit(cross_val=False)
    model.summary(assumptions=True, performance=True, plots=['forest_plot'])

    print("✓ Univariate model tested")
    return model


def test_complete_pipeline():
    print("\n" + "=" * 80)
    print("TEST 7: Complete Pipeline")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(cross_val=True, n_splits=5)

    model.summary(
        assumptions=True,
        performance=True,
        cross_val=True,
        plots=['forest_plot', 'roc_curve', 'confusion_matrix'],
        vif_threshold=5.0
    )

    # Removed: Don't call internal methods - plots already shown above
    print("✓ Complete pipeline executed")
    return model


def test_with_labels():
    """New test to verify custom variable labels work correctly."""
    print("\n" + "=" * 80)
    print("TEST 8: Custom Variable Labels")
    print("=" * 80)

    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']

    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic Blood Pressure (mmHg)'
    }

    model = RAPID_LogisticRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors
    )

    model.fit(labels=labels, cross_val=False)
    model.summary(assumptions=True, performance=True, plots=['forest_plot'])

    print("✓ Custom labels applied successfully")
    return model

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    tests = [
        test_basic_fit,
        test_assumptions,
        test_performance_metrics,
        test_cross_validation,
        test_plots,
        test_univariate,
        test_complete_pipeline,
        test_with_labels
    ]

    results = {}

    for test in tests:
        try:
            test()
            results[test.__name__] = "PASS"
        except Exception as e:
            results[test.__name__] = f"FAIL: {e}"
            print(f"✗ {test.__name__} failed: {e}")

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for k, v in results.items():
        status = '✓' if v == 'PASS' else '✗'
        print(f"{status} {k}: {v}")

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()