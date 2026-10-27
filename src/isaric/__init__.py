"""
ISARIC RAPID: Reusable Analytical Pipelines for Infectious Diseases.

This package implements the RAPID methodology for clinical research
on infectious diseases. It provides a unified interface for data
cleaning, preprocessing, modeling, evaluation, validation, and
visualization.

Quick Start:
    import isaric as isa

    # Data Preparation
    df_clean = isa.Clean(remove_zero_variance=True).execute(df)

    # Analytics
    model = isa.RAPID.create(data=df, model="logistic", ...)
    model.fit()
    model.summary()

Modules:
    - datacleaning: Step 1 - Data Cleaning
    - preprocessing: Step 2 - Data Preprocessing
    - modeling: Step 3 - Modelling
    - modelevaluation: Step 4 - Model Evaluation
    - validation: Step 5 - Validation
    - visualization: Step 6 - Visualization
"""

__version__ = "0.2.0"

# ============================================================================
# CORE CLASSES
# ============================================================================

from isaric.rapid import RAPID
from isaric.parser import (
    validate_arc_format,
    parse_to_arc_format,
    prepare_data_for_rapid
)

# ============================================================================
# DATA CLEANING (Step 1)
# ============================================================================

from isaric.datacleaning.clean import Clean
from isaric.datacleaning.duplicates import (
    exact_match_removal,
    key_based_deduplication
)
from isaric.datacleaning.harmonise_units import (
    linear_conversion,
    lookup_tables,
    convert_temperature_celsius_to_fahrenheit,
    convert_temperature_fahrenheit_to_celsius,
    convert_weight_kg_to_lbs,
    convert_weight_lbs_to_kg
)
from isaric.datacleaning.remove_zero_variance import (
    frequency_ratio_analysis,
    unique_value_count,
    get_zero_variance_features,
    get_near_zero_variance_features
)
from isaric.datacleaning.handle_missing import (
    drop_rows,
    drop_columns,
    impute_mean,
    impute_median,
    impute_mode
)

# ============================================================================
# DATA PREPROCESSING (Step 2)
# ============================================================================

from isaric.preprocessing.preprocess import Preprocess
from isaric.preprocessing.datasplitting import (
    simple_random_split,
    stratified_split,
    temporal_split
)
from isaric.preprocessing.imputation import (
    mice_imputation
)
from isaric.preprocessing.collinearity import (
    vif_analysis,
    get_vif_table,
    pearson_correlation,
    get_correlation_pairs
)
from isaric.preprocessing.normalization import (
    standardize,
    minmax_scale
)
from isaric.preprocessing.encoding import (
    onehot_encode,
    label_encode,
    target_encode
)
from isaric.preprocessing.scaling import (
    log_transform,
    boxcox_transform
)
from isaric.preprocessing.featureselection import (
    variance_threshold,
    lasso_selection,
    rfe_selection,
    filter_selection
)
from isaric.preprocessing.temporalencoding import (
    duration_encode,
    cyclical_encode
)

# ============================================================================
# MODELING (Step 3) - SUBCLASSES
# ============================================================================

from isaric.modeling.regression import LogisticRegression, GLM
from isaric.modeling.survival import Survival
from isaric.modeling.clustering import LCA, KMeans
from isaric.modeling.descriptive import Descriptive
from isaric.modeling.treebased import (
    DecisionTree,
    RandomForest,
    XGBoost,
    LightGBM,
    CatBoost
)
from isaric.modeling.predictive import (
    Lasso,
    Ridge,
    ElasticNet,
    SVM,
    LogisticL2
)

# ============================================================================
# MODEL EVALUATION (Step 4)
# ============================================================================

from isaric.modelevaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
    compute_information_criteria,
    compute_pseudo_r2,
    compute_survival_metrics,
    compute_calibration_metrics,
    compute_clustering_metrics,
    select_classification_threshold
)
from isaric.modelevaluation.crossvalidation import (
    kfold_cross_validation,
    repeated_kfold_cross_validation,
    stratified_kfold_cross_validation,
    out_of_fold_predictions,
    build_repeated_stratified_kfold
)
from isaric.modelevaluation.calibration import (
    calibration_curve,
    binned_calibration,
    compute_brier_score,
    predicted_vs_observed,
    survival_calibration,
    survival_brier_score,
    survival_roc,
    residuals_vs_fitted,
    qq_plot
)
from isaric.modelevaluation.traintest import (
    holdout_validation,
    stratified_holdout,
    temporal_holdout,
    temporal_train_test_split
)
from isaric.modelevaluation.assumptions import (
    test_durbin_watson,
    test_shapiro_wilk,
    test_vif,
    test_cooks_distance,
    test_epv,
    test_proportional_hazards,
    likelihood_ratio_test,
    test_binned_residuals,

)
from isaric.modelevaluation.gridsearch import (
    run_grid_search,
    run_custom_grid_search,
    run_clustering_grid_search,
    bootstrap_lrt_lca,
    bootstrap_stepmix,
    bootstrap_lrt_stepmix,
    bootstrap_ci_stepmix,
    bootstrap_selection_rate,
)
from isaric.modelevaluation.bootstrap import (
    non_parametric_bootstrap,
    confidence_interval,
    bootstrap_metrics,
    bootstrap_c_index,
    bootstrap_coefficients,
    bootstrap_hazard_ratios,
)

# ============================================================================
# VALIDATION (Step 5)
# ============================================================================

from isaric.validation.external import (
    temporal_validation,
    geographic_validation,
    recalibration
)
from isaric.validation.sensitivity import (
    alternative_missing_handling,
    outlier_variation,
    outcome_variation
)
from isaric.validation.subgroup import (
    stratified_metrics,
    stratified_regression,
    interaction_test
)
from isaric.validation.netprofit import (
    decision_curve_analysis,
    net_benefit_curve,
    treat_all_none,
    clinical_utility_curve
)

# ============================================================================
# VISUALIZATION (Step 6)
# ============================================================================

from isaric.visualization.barplots import (
    simple_bar_plot,
    grouped_bar_plot,
    stacked_bar_plot,
    histogram_plot,  
    boxplot,         
)
from isaric.visualization.lineplots import (
    time_series_plot,
    multi_line_plot,
    line_with_ci,
    binned_residuals_plot,
    information_criteria_plot,
    grid_search_metric_plot,
    selection_rate_plot,
    bootstrap_ci_plot, 
)
from isaric.visualization.upsetplots import (
    upset_plot,
    set_size_plot,
    intersection_size_plot,
)
from isaric.visualization.survivalcurves import (
    kaplan_meier_curve,
    compare_survival_curves,
    baseline_survival_curve,
    schoenfeld_residuals_plot,
    martingale_residuals_plot,
    deviance_residuals_plot,
    brier_score_plot,
    roc_time_dependent_plot,
    calibration_plot
)
from isaric.visualization.heatmaps import (
    correlation_heatmap,
    confusion_matrix_heatmap,
    lca_profile_heatmap,
    pairplot,
    crosstab_heatmap,
)
from isaric.visualization.forestplots import (
    odds_ratio_plot,
    hazard_ratio_plot,
    coefficient_plot
)
from isaric.visualization.sankey import (
    patient_pathway,
    cohort_flow
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version
    "__version__",

    # Core
    "RAPID",
    "validate_arc_format",
    "parse_to_arc_format",
    "prepare_data_for_rapid",

    # Data Cleaning
    "Clean",
    "exact_match_removal",
    "key_based_deduplication",
    "linear_conversion",
    "lookup_tables",
    "convert_temperature_celsius_to_fahrenheit",
    "convert_temperature_fahrenheit_to_celsius",
    "convert_weight_kg_to_lbs",
    "convert_weight_lbs_to_kg",
    "frequency_ratio_analysis",
    "unique_value_count",
    "get_zero_variance_features",
    "get_near_zero_variance_features",
    "drop_rows",
    "drop_columns",
    "impute_mean",
    "impute_median",
    "impute_mode",

    # Data Preprocessing
    "Preprocess",
    "simple_random_split",
    "stratified_split",
    "temporal_split",
    "mice_imputation",
    "vif_analysis",
    "get_vif_table",
    "pearson_correlation",
    "get_correlation_pairs",
    "standardize",
    "minmax_scale",
    "onehot_encode",
    "label_encode",
    "target_encode",
    "log_transform",
    "boxcox_transform",
    "variance_threshold",
    "lasso_selection",
    "rfe_selection",
    "filter_selection",
    "duration_encode",
    "cyclical_encode",

    # Modeling - Subclasses
    "LogisticRegression",
    "GLM",
    "Survival",
    "LCA",
    "KMeans",
    "Descriptive",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Lasso",
    "Ridge",
    "ElasticNet",
    "SVM",
    "LogisticL2",

    # Model Evaluation
    "compute_classification_metrics",
    "compute_regression_metrics",
    "compute_information_criteria",
    "compute_pseudo_r2",
    "compute_survival_metrics",
    "compute_calibration_metrics",
    "compute_clustering_metrics",
    "select_classification_threshold",
    "kfold_cross_validation",
    "repeated_kfold_cross_validation",
    "stratified_kfold_cross_validation",
    "out_of_fold_predictions",
    "build_repeated_stratified_kfold",
    "calibration_curve",
    "binned_calibration",
    "compute_brier_score",
    "predicted_vs_observed",
    "survival_calibration",
    "survival_brier_score",
    "survival_roc",
    "residuals_vs_fitted",
    "qq_plot",
    "holdout_validation",
    "stratified_holdout",
    "temporal_holdout",
    "temporal_train_test_split",
    "test_durbin_watson",
    "test_shapiro_wilk",
    "test_vif",
    "test_cooks_distance",
    "test_epv",
    "test_proportional_hazards",
    "likelihood_ratio_test",
    "run_grid_search",
    "run_custom_grid_search",
    "run_clustering_grid_search",
    "test_binned_residuals",
    "bootstrap_lrt_lca",
    "bootstrap_stepmix",
    "bootstrap_lrt_stepmix",
    "bootstrap_ci_stepmix",
    "bootstrap_ci_plot",
    "bootstrap_selection_rate",
    "bootstrap_c_index",
    "bootstrap_coefficients",
    "bootstrap_hazard_ratios",

    # Validation
    "temporal_validation",
    "geographic_validation",
    "recalibration",
    "non_parametric_bootstrap",
    "confidence_interval",
    "bootstrap_metrics",
    "alternative_missing_handling",
    "outlier_variation",
    "outcome_variation",
    "stratified_metrics",
    "stratified_regression",
    "interaction_test",
    "decision_curve_analysis",
    "net_benefit_curve",
    "treat_all_none",
    "clinical_utility_curve",

    # Visualization
    "simple_bar_plot",
    "grouped_bar_plot",
    "stacked_bar_plot",
    "time_series_plot",
    "multi_line_plot",
    "line_with_ci",
    "upset_plot",
    "set_size_plot",
    "intersection_size_plot",
    "kaplan_meier_curve",
    "compare_survival_curves",
    "baseline_survival_curve",
    "schoenfeld_residuals_plot",
    "martingale_residuals_plot",
    "deviance_residuals_plot",
    "brier_score_plot",
    "roc_time_dependent_plot",
    "calibration_plot",
    "correlation_heatmap",
    "confusion_matrix_heatmap",
    "lca_profile_heatmap",
    "odds_ratio_plot",
    "hazard_ratio_plot",
    "coefficient_plot",
    "patient_pathway",
    "cohort_flow",
    "binned_residuals_plot",
    "information_criteria_plot",
    "pairplot",
    "histogram_plot",
    "boxplot",
    "grid_search_metric_plot",
    "crosstab_heatmap",
    "selection_rate_plot",
]