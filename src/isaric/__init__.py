# isaric.py — ISARIC 3.0 RAPID API

# Importação dos submódulos
from isaric.datacleaning import (
    duplicatehandling, 
    harmonisingunits, 
    valuesmissing, 
    zerovariance
)

from isaric.preprocessing import (
    collinearity, 
    datasplitting, 
    featureselection, 
    imputation, 
    normalization, 
    onehotencoding, 
    scaling, 
    temporalencoding
)

from isaric.modeling import (
    clustering, 
    descriptivestats, 
    regression, 
    survival, 
    treebased
)

from isaric.modelevaluation import (
    calibration, 
    crossvalidation, 
    metrics, 
    traintest
)

from isaric.validation import (
    bootstrap, 
    external, 
    netprofit, 
    sensitivity, 
    subgroup
)

from isaric.visualization import (
    barplots, 
    forestplots, 
    heatmaps, 
    lineplots, 
    sankey, 
    survivalcurves
)

# Fachadas por etapa
class DataCleaning:
    def duplicatehandling(self, df, method="iqr"):
        return duplicatehandling.remove_duplicates(df)

    def harmonise_units(self, df):
        return harmonisingunits.harmonise_units(df)

    def handle_missing_values(self, df, strategy="mean"):
        return valuesmissing.remove_duplicates(df, strategy)

    def remove_zero_variance(self, df):
        return zerovariance.remove_zero_variance_features(df)

class Preprocessing:
    def split_data(self, df, test_size=0.2, random_state=42):
        return datasplitting.split_data(df, test_size, random_state)

    def impute_missing(self, df, strategy="mean"):
        return imputation.impute_missing(df, strategy)

    def detect_collinearity(self, df, threshold=0.9):
        return collinearity.detect_collinearity(df, threshold)

    def normalize(self, df):
        return normalization.noralize(df)

    def encode_categoricals(self, df):
        return onehotencoding.encode_categoricals(df)

    def standardize(self, df):
        return scaling.standardize(df)

    def select_features(self, df, target):
        return featureselection.select_features_lasso((df, target))

    def encode_temporal(self, df):
        return temporalencoding.encode_temporal(df)

class Modeling:
    def generate_summary_statistics(self, df):
        return descriptivestats.generate_summary_statistics(df)

    def fit_linear_regression(self, df, target):
        return regression.fit_linear_regression(df, target)

    def fit_random_forest(self, df, target):
        return treebased.fit_random_forest(df, target)

    def fit_cox_model(self, df, duration_col, event_col):
        return survival.fit_cox_model(df, duration_col, event_col)

    def apply_kmeans(self, df, n_clusters=3):
        return clustering.apply_kmeans(df, n_clusters)

class ModelEvaluation:
    def train_test_split_evaluation(self, df, target):
        return traintest.train_test_split_evaluation(df, target)

    def perform_cross_validation(self, model, X, y, cv=5):
        return crossvalidation.perform_cross_validation(model, X, y, cv)

    def compute_classification_metrics(self, y_true, y_pred):
        return metrics.compute_classification_metrics(y_true, y_pred)

    def compute_calibration_curve(self, y_true, y_prob):
        return calibration.compute_calibration_curve(y_true, y_prob)

class Validation:
    def validate_with_external_dataset(self, model, external_df):
        return external.validate_with_external_dataset(model, external_df)

    def bootstrap_validation(self, model, df, target, n_iterations=1000):
        return bootstrap.bootstrap_validation(model, df, target, n_iterations)

    def sensitivity_analysis(self, df, variable):
        return sensitivity.sensitivity_analysis(df, variable)

    def subgroup_analysis(self, df, subgroup_col):
        return subgroup.subgroup_analysis(df, subgroup_col)

    def calculate_net_profit(self, df, cost_col, revenue_col):
        return netprofit.calculate_net_profit(df, cost_col, revenue_col)

class Visualization:
    def plot_line(self, df, x, y):
        return lineplots.plot_line(df, x, y)

    def plot_bar(self, df, x, y):
        return barplots.plot_bar(df, x, y)

    def plot_upset(self, df, sets):
        return sankey.plot_upset(df, sets)

    def plot_survival_curve(self, df, duration_col, event_col):
        return survivalcurves.plot_survival_curve(df, duration_col, event_col)

    def plot_heatmap(self, df):
        return heatmaps.plot_heatmap(df)

    def plot_forest(self, df, effect_col, ci_lower_col, ci_upper_col):
        return forestplots.plot_forest(df, effect_col, ci_lower_col, ci_upper_col)

    def plot_sankey(self, df, source_col, target_col, value_col):
        return sankey.plot_sankey(df, source_col, target_col, value_col)

# Exposição pública da API
datacleaning = DataCleaning()
preprocessing = Preprocessing()
modeling = Modeling()
modelevaluation = ModelEvaluation()
validation = Validation()
visualization = Visualization()
