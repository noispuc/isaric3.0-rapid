"""
Clustering models for the RAPID methodology.

This module provides functions to configure clustering models and
concrete pipeline classes for Latent Class Analysis (LCA) and
K-Means clustering.

Functions (Configuration):
- create_lca_model: Configure Latent Class Analysis (LCA).
- create_kmeans_model: Configure K-Means clustering.

Subclasses (Pipelines):
- LCA: Concrete pipeline for Latent Class Analysis.
- KMeans: Concrete pipeline for K-Means clustering.

Helper Functions:
- _build_result_df: Build class profiles DataFrame for LCA.
- _validate_binary_columns: Validate that columns are binary (0/1).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.cluster import KMeans as SklearnKMeans
from stepmix.stepmix import StepMix
from isaric.rapid import RAPID


# ============================================================================
# PUBLIC FUNCTIONS (CONFIGURATION)
# ============================================================================

def create_lca_model(
    data: pd.DataFrame,
    measurement_vars: List[str],
    n_components: int = 3,
    n_init: int = 5,
    max_iter: int = 500,
    random_state: int = 42
) -> Tuple[StepMix, pd.DataFrame]:
    """
    Configure a Latent Class Analysis (LCA) model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        measurement_vars: List of binary variables.
        n_components: Number of latent classes (default 3).
        n_init: Number of random initializations.
        max_iter: Maximum EM iterations.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (model, X).

    Raises:
        ValueError: If columns are not binary or missing.
    """
    _validate_binary_columns(data, measurement_vars)

    X = data[measurement_vars].copy()

    model = StepMix(
        n_components=n_components,
        measurement="bernoulli",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    return model, X


def create_kmeans_model(
    data: pd.DataFrame,
    predictors: List[str],
    n_clusters: int = 3,
    random_state: int = 42
) -> Tuple[SklearnKMeans, pd.DataFrame]:
    """
    Configure a K-Means clustering model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        predictors: List of numeric variables.
        n_clusters: Number of clusters (default 3).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (model, X).

    Raises:
        ValueError: If columns are not numeric or missing.
    """
    for col in predictors:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(data[col]):
            raise ValueError(f"Column '{col}' must be numeric for K-Means.")

    if n_clusters < 1:
        raise ValueError(f"n_clusters must be at least 1. Received: {n_clusters}")

    X = data[predictors].copy()

    model = SklearnKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    return model, X


# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _build_result_df(
    model: StepMix,
    X: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Build class profiles DataFrame from fitted LCA model.

    Args:
        model: Fitted StepMix model.
        X: Binary matrix used for fitting.
        feature_names: Variable names.

    Returns:
        DataFrame with shape (n_classes, n_variables).
    """
    X_values = X.values if isinstance(X, pd.DataFrame) else X
    posteriors = model.predict_proba(X_values)

    n_samples, n_features = X_values.shape
    n_classes = posteriors.shape[1]
    profiles = np.zeros((n_classes, n_features), dtype=float)

    for c in range(n_classes):
        w = posteriors[:, c].reshape(-1, 1)
        denom = w.sum()
        if denom <= 0:
            profiles[c, :] = np.nan
        else:
            profiles[c, :] = (w * X_values).sum(axis=0) / denom

    prof_df = pd.DataFrame(profiles, columns=feature_names)
    prof_df.index = [f"Class_{i+1}" for i in range(n_classes)]
    return prof_df


def _validate_binary_columns(
    data: pd.DataFrame,
    columns: List[str]
) -> None:
    """
    Validate that columns are binary (0/1).

    Args:
        data: Input DataFrame.
        columns: Column names to validate.

    Raises:
        ValueError: If columns are not found or not binary.
    """
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        unique_values = data[col].dropna().unique()
        valid_binary = {0, 1, 0.0, 1.0, True, False}

        if not set(unique_values).issubset(valid_binary):
            raise ValueError(
                f"Column '{col}' must be binary (0/1). "
                f"Found: {unique_values}"
            )


# ============================================================================
# SUBCLASSES (INHERIT FROM RAPID)
# ============================================================================

class LCA(RAPID):
    """
    Concrete pipeline for Latent Class Analysis.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: StepMix,
        X: pd.DataFrame,
        measurement_vars: List[str],
        n_components: int = 3,
        **kwargs
    ):
        """
        Initialize LCA with configured model and data.

        Args:
            model: Configured StepMix (from create_lca_model).
            X: Binary matrix for training.
            measurement_vars: Binary variable names.
            n_components: Number of latent classes.
        """
        self._model = model
        self.X = X
        self.measurement_vars = measurement_vars
        self.n_components = n_components
        self.model_type = "lca"
        self.y = None
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for LCA."""
        self.plots_map = {
            "profile_heatmap": self._profile_heatmap,
            "cluster_distribution": self._cluster_distribution,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "lca",
        measurement_vars: Optional[List[str]] = None,
        n_components: int = 3,
        n_init: int = 5,
        max_iter: int = 500,
        random_state: int = 42,
        **params
    ) -> "LCA":
        """
        Configure and instantiate the LCA pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "lca").
            measurement_vars: Binary variable names.
            n_components: Number of latent classes.
            n_init: Number of random initializations.
            max_iter: Maximum EM iterations.
            random_state: Seed for reproducibility.

        Returns:
            LCA instance ready for training.
        """
        model_config, X = create_lca_model(
            data=data,
            measurement_vars=measurement_vars,
            n_components=n_components,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )

        return cls(
            model=model_config,
            X=X,
            measurement_vars=measurement_vars,
            n_components=n_components,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self):
        """Train the LCA model."""
        return self._model.fit(self.X)

    def _build_result_df(self):
        """Build class profiles DataFrame."""
        return _build_result_df(
            model=self.fitted_model,
            X=self.X,
            feature_names=self.measurement_vars
        )

    def _calculate_metrics(self, metrics=None):
        """Calculate clustering metrics."""
        from isaric.modelevaluation.metrics import compute_clustering_metrics
        return compute_clustering_metrics(
            self.fitted_model,
            self.X,
            self.y
        )

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Not applicable for LCA."""
        return None

    def _calibration_curve(self):
        """Not applicable for LCA."""
        return None

    def _check_assumptions(self):
        """Not applicable for LCA."""
        return None

    def _train_test_split(self, test_size=0.2):
        """Not applicable for LCA."""
        return None

    def _validate_external(self, external_data):
        """Not applicable for LCA."""
        return None

    def _validate_bootstrap(self, n_iterations=1000):
        """Bootstrap validation for class profiles."""
        return None

    def _validate_sensitivity(self):
        """Not applicable for LCA."""
        return None

    def _validate_subgroups(self, subgroups):
        """Not applicable for LCA."""
        return None

    def _validate_net_benefit(self):
        """Not applicable for LCA."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _profile_heatmap(self, backend="plotly"):
        """Generate LCA class profiles heatmap."""
        from isaric.visualization.heatmaps import lca_profile_heatmap

        return lca_profile_heatmap(
            self.result_df,
            n_components=self.n_components,
            title="LCA Class Profiles",
            backend=backend
        )

    def _cluster_distribution(self, backend="plotly"):
        """Generate cluster distribution bar plot."""
        from isaric.visualization.barplots import simple_bar_plot

        cluster_labels = self.fitted_model.predict(self.X)
        cluster_counts = pd.Series(cluster_labels).value_counts()
        cluster_df = pd.DataFrame({
            'Class': [f'Class_{i+1}' for i in range(len(cluster_counts))],
            'Count': cluster_counts.values
        })

        return simple_bar_plot(
            cluster_df,
            x_col='Class',
            y_col='Count',
            title="Latent Class Distribution",
            backend=backend
        )


class KMeans(RAPID):
    """
    Concrete pipeline for K-Means clustering.

    Implements create() (abstract from RAPID). Inherits concrete methods:
    fit(), summary(), save(), validation(), report(), decide().
    """

    def __init__(
        self,
        model: SklearnKMeans,
        X: pd.DataFrame,
        predictors: List[str],
        n_clusters: int = 3,
        **kwargs
    ):
        """
        Initialize KMeans with configured model and data.

        Args:
            model: Configured SklearnKMeans (from create_kmeans_model).
            X: Numeric matrix for training.
            predictors: Numeric variable names.
            n_clusters: Number of clusters.
        """
        self._model = model
        self.X = X
        self.predictors = predictors
        self.n_clusters = n_clusters
        self.model_type = "kmeans"
        self.y = None
        self.fitted_model = None
        self.result_df = None
        self.metrics = None
        self.plots_map = {}

        self._setup_plots_map()

        super().__init__()

    def _setup_plots_map(self):
        """Configure available plots for KMeans."""
        self.plots_map = {
            "cluster_distribution": self._cluster_distribution,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "kmeans",
        predictors: Optional[List[str]] = None,
        n_clusters: int = 3,
        random_state: int = 42,
        **params
    ) -> "KMeans":
        """
        Configure and instantiate the KMeans pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "kmeans").
            predictors: Numeric variable names.
            n_clusters: Number of clusters.
            random_state: Seed for reproducibility.

        Returns:
            KMeans instance ready for training.
        """
        model_config, X = create_kmeans_model(
            data=data,
            predictors=predictors,
            n_clusters=n_clusters,
            random_state=random_state
        )

        return cls(
            model=model_config,
            X=X,
            predictors=predictors,
            n_clusters=n_clusters,
            **params
        )

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self):
        """Train the K-Means model."""
        return self._model.fit(self.X)

    def _build_result_df(self):
        """Build cluster centers DataFrame."""
        centers = self.fitted_model.cluster_centers_
        result_df = pd.DataFrame(
            centers,
            columns=self.predictors
        )
        result_df.index = [f'Cluster_{i}' for i in range(len(centers))]
        return result_df

    def _calculate_metrics(self, metrics=None):
        """Calculate clustering metrics."""
        from isaric.modelevaluation.metrics import compute_clustering_metrics
        return compute_clustering_metrics(
            self.fitted_model,
            self.X,
            None
        )

    def _cross_validate(self, k_folds=5, repetitions=1):
        """Not applicable for KMeans."""
        return None

    def _calibration_curve(self):
        """Not applicable for KMeans."""
        return None

    def _check_assumptions(self):
        """Not applicable for KMeans."""
        return None

    def _train_test_split(self, test_size=0.2):
        """Not applicable for KMeans."""
        return None

    def _validate_external(self, external_data):
        """Not applicable for KMeans."""
        return None

    def _validate_bootstrap(self, n_iterations=1000):
        """Not applicable for KMeans."""
        return None

    def _validate_sensitivity(self):
        """Not applicable for KMeans."""
        return None

    def _validate_subgroups(self, subgroups):
        """Not applicable for KMeans."""
        return None

    def _validate_net_benefit(self):
        """Not applicable for KMeans."""
        return None

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _cluster_distribution(self, backend="plotly"):
        """Generate cluster distribution bar plot."""
        from isaric.visualization.barplots import simple_bar_plot

        cluster_counts = pd.Series(self.fitted_model.labels_).value_counts()
        cluster_df = pd.DataFrame({
            'Cluster': [f'Cluster_{i}' for i in cluster_counts.index],
            'Count': cluster_counts.values
        })

        return simple_bar_plot(
            cluster_df,
            x_col='Cluster',
            y_col='Count',
            title="K-Means Cluster Distribution",
            backend=backend
        )