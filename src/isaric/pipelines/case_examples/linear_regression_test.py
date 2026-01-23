"""
Testing script for RAPID_LinearRegression module.
Tests all functionality including fit, assumptions, performance metrics, and plots.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from isaric.pipelines.linear_regression import RAPID_LinearRegression

# ============================================================================
# GENERATE SYNTHETIC DATA
# ============================================================================

def generate_test_data(n_samples=200, random_state=42):
    """
    Generate synthetic data for linear regression testing.
    Includes both continuous and categorical predictors.
    """
    np.random.seed(random_state)
    
    # Continuous predictors
    age = np.random.normal(50, 15, n_samples)
    bmi = np.random.normal(25, 5, n_samples)
    blood_pressure = np.random.normal(120, 20, n_samples)
    
    # Categorical predictors
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
    
    # Create DataFrame
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
# TEST FUNCTIONS
# ============================================================================

def test_basic_fit():
    """Test basic model fitting."""
    print("=" * 80)
    print("TEST 1: Basic Model Fitting")
    print("=" * 80)
    
    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex', 'smoking_status']
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    # Fit without cross-validation to speed up test
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
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    model.fit(labels=labels, cross_val=False)
    
    print("\n✓ Model fitted with custom labels")
    print("\nSummary DataFrame:")
    print(model.summary_df)
    
    return model

def test_assumptions():
    """Test assumption checking."""
    print("\n" + "=" * 80)
    print("TEST 3: Assumption Tests")
    print("=" * 80)
    
    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    # Test all assumptions
    print("\n--- ASSUMPTION TEST RESULTS ---\n")
    model.summary(assumptions=True, performance=False, plots=None)
    
    print("\n✓ All assumption tests completed")
    
    return model

def test_performance_metrics():
    """Test performance metrics."""
    print("\n" + "=" * 80)
    print("TEST 4: Performance Metrics")
    print("=" * 80)
    
    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- PERFORMANCE METRICS ---\n")
    model.summary(assumptions=False, performance=True, plots=None)
    
    print("\n✓ Performance metrics calculated")
    
    return model

def test_cross_validation():
    """Test cross-validation."""
    print("\n" + "=" * 80)
    print("TEST 5: Cross-Validation")
    print("=" * 80)
    
    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure']
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    # Fit with cross-validation
    model.fit(cross_val=True, n_splits=5)
    
    print("\n--- CROSS-VALIDATION RESULTS ---\n")
    model.summary(assumptions=False, performance=False, cross_val=True, plots=None)
    
    print("\n✓ Cross-validation completed")
    
    return model

def test_plots():
    """Test all plotting functions."""
    print("\n" + "=" * 80)
    print("TEST 6: Plotting Functions")
    print("=" * 80)
    
    df = generate_test_data()
    predictors = ['age', 'bmi', 'blood_pressure', 'sex']
    
    labels = {
        'age': 'Age (years)',
        'bmi': 'Body Mass Index',
        'blood_pressure': 'Systolic BP (mmHg)',
        'sex': 'Sex'
    }
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    model.fit(labels=labels, cross_val=False)
    
    # Generate all plots
    print("\n--- GENERATING PLOTS ---\n")
    
    plots_to_generate = ['forest_plot', 'residuals_vs_fitted', 'qq_plot']
    
    for plot_name in plots_to_generate:
        print(f"Generating {plot_name}...")
        model.summary(assumptions=False, performance=False, plots=[plot_name])
    
    print("\n✓ All plots generated successfully")
    
    return model

def test_univariate():
    """Test univariate regression."""
    print("\n" + "=" * 80)
    print("TEST 7: Univariate Regression")
    print("=" * 80)
    
    df = generate_test_data()
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=['bmi'],
        regression_type='Uni'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- UNIVARIATE RESULTS ---")
    print("\nSummary DataFrame:")
    print(model.summary_df)
    
    print("\n--- PERFORMANCE METRICS ---\n")
    model.summary(assumptions=False, performance=True, plots=None)
    
    print("\n✓ Univariate regression completed")
    
    return model

def test_multicollinearity():
    """Test VIF for multicollinearity detection."""
    print("\n" + "=" * 80)
    print("TEST 8: Multicollinearity Detection (VIF)")
    print("=" * 80)
    
    df = generate_test_data(n_samples=300)
    
    # Add highly correlated variables
    df['bmi_squared'] = df['bmi'] ** 2
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    predictors = ['age', 'bmi', 'bmi_squared', 'age_bmi_interaction', 'blood_pressure']
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    model.fit(cross_val=False)
    
    print("\n--- VIF TEST (with threshold = 5.0) ---\n")
    model.summary(assumptions=True, performance=False, plots=None, vif_threshold=5.0)
    
    print("\n--- VIF TEST (with threshold = 10.0) ---\n")
    model.summary(assumptions=True, performance=False, plots=None, vif_threshold=10.0)
    
    print("\n✓ Multicollinearity test completed")
    
    return model

def test_complete_pipeline():
    """Test complete analysis pipeline."""
    print("\n" + "=" * 80)
    print("TEST 9: Complete Analysis Pipeline")
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
    
    model = RAPID_LinearRegression(
        data=df,
        outcome_str='outcome',
        predictors_list=predictors,
        regression_type='Multi'
    )
    
    # Fit with all options
    print("\nFitting model with cross-validation...")
    model.fit(labels=labels, cross_val=True, n_splits=5)
    
    # Full summary report
    print("\n" + "=" * 80)
    print("COMPLETE MODEL SUMMARY")
    print("=" * 80)
    
    model.summary(
        assumptions=True,
        performance=True,
        cross_val=True,
        plots=['forest_plot', 'residuals_vs_fitted', 'qq_plot'],
        vif_threshold=5.0
    )
    
    print("\n✓ Complete pipeline executed successfully")
    
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
        test_basic_fit,
        test_with_labels,
        test_assumptions,
        test_performance_metrics,
        test_cross_validation,
        test_plots,
        test_univariate,
        test_multicollinearity,
        test_complete_pipeline
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