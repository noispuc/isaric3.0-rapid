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
    n_clusters: int = 3,
    n_init: int = 5,
    max_iter: int = 500,
    random_seed=42
) -> Tuple[StepMix, pd.DataFrame]:
    """
    Configure a Latent Class Analysis (LCA) model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        measurement_vars: List of binary variables.
        n_clusters: Number of latent classes (default 3).
        n_init: Number of random initializations.
        max_iter: Maximum EM iterations.
        random_seed: Seed for reproducibility.

    Returns:
        Tuple of (model, X).

    Raises:
        ValueError: If columns are not binary or missing.
    """
    _validate_binary_columns(data, measurement_vars)

    X = data[measurement_vars].copy()

    model = StepMix(
        n_components=n_clusters,
        measurement="bernoulli",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_seed,
        verbose=0,
    )

    return model, X


def create_kmeans_model(
    data: pd.DataFrame,
    predictors: List[str],
    n_clusters: int = 3,
    random_seed=42
) -> Tuple[SklearnKMeans, pd.DataFrame]:
    """
    Configure a K-Means clustering model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        predictors: List of numeric variables.
        n_clusters: Number of clusters (default 3).
        random_seed: Seed for reproducibility.

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
        random_state=random_seed,
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
        n_clusters: int = 3,
        **kwargs
    ):
        """
        Initialize LCA with configured model and data.

        Args:
            model: Configured StepMix (from create_lca_model).
            X: Binary matrix for training.
            measurement_vars: Binary variable names.
            n_clusters: Number of latent classes.
        """
        self._model = model
        self.X = X
        self.measurement_vars = measurement_vars
        self.n_clusters = n_clusters
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
            "radar_plot": self._radar_plot,
            "information_criteria": self._information_criteria,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "lca",
        measurement_vars: Optional[List[str]] = None,
        n_clusters: int = 3,
        n_init: int = 5,
        max_iter: int = 500,
        random_seed: int = 42,
        **params
    ) -> "LCA":
        """
        Configure and instantiate the LCA pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "lca").
            measurement_vars: Binary variable names.
            n_clusters: Number of latent classes.
            n_init: Number of random initializations.
            max_iter: Maximum EM iterations.
            random_seed: Seed for reproducibility.

        Returns:
            LCA instance ready for training.
        """
        model_config, X = create_lca_model(
            data=data,
            measurement_vars=measurement_vars,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_seed=random_seed
        )

        instance = cls(
            model=model_config,
            X=X,
            measurement_vars=measurement_vars,
            n_clusters=n_clusters,
            **params
        )
        instance._random_seed = random_seed
        
        return instance

    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train LCA model, optionally with class enumeration grid search.

        When grid_search=True, trains ALL k in the range and stores
        each fitted model in self.model_objects (dict k → StepMix).
        The user can later explore alternatives via summary(k=N).

        The selection_metric determines which k is used as the DEFAULT
        (self.fitted_model). It does NOT prevent exploration of others.

        Args:
            grid_search: Enable class enumeration.
            param_grid: {"n_components": [2, 3, 4, 5]} or None (default 2..10).
            selection_metric: "BIC", "AIC", "CAIC", "SABIC", or "auto" (=BIC).

        Returns:
            Fitted StepMix model (the default one, per selection_metric).
        """
        
        # Auto-ativa grid_search se param_grid foi passado
        if not grid_search and param_grid is not None:
            grid_search = True

        # LCA exige grid search ou n_components explícito
        if not grid_search and param_grid is None:
            raise ValueError(
                "LCA requires class enumeration. "
                "Pass either `grid_search=True` or "
                "`param_grid={'n_components': [...]}`."
            )
        
        # ============ GRID SEARCH ============
        if param_grid is None:
            param_grid = {"n_clusters": list(range(2, 11))}

        if selection_metric == "auto":
            selection_metric = "BIC"

        n_samples = self.X.shape[0]
        results = []
        model_objects = {}
        best_score = np.inf
        best_k = self.n_clusters

        for k in param_grid.get("n_clusters", [self.n_clusters]):
            model = StepMix(
                n_components=k,
                measurement="bernoulli",
                n_init=5,
                max_iter=500,
                random_state=self._random_seed,
                verbose=0,
            )
            model.fit(self.X)

            avg_ll = model.score(self.X)
            ll = avg_ll * n_samples
            npar = model.n_parameters
            bic = -2 * ll + npar * np.log(n_samples)
            aic = -2 * ll + 2 * npar
            caic = -2 * ll + npar * (np.log(n_samples) + 1)
            sabic = -2 * ll + npar * np.log(n_samples * ((n_samples + 2) / 24))
            entropy = model.entropy(self.X)

            results.append({
                'n_clusters': k,
                'LL': ll,
                'AIC': aic,
                'BIC': bic,
                'CAIC': caic,
                'SABIC': sabic,
                'entropy': entropy,
                'converged': model.converged_,
            })

            # Guarda o modelo ajustado deste k
            model_objects[k] = model

            # Verifica se é o melhor pelo selection_metric
            metric_value = {
                'BIC': bic, 'AIC': aic, 'CAIC': caic, 'SABIC': sabic
            }.get(selection_metric, bic)

            if metric_value < best_score:
                best_score = metric_value
                best_k = k

        # ============ GUARDA RESULTADOS ============
        self.grid_results = pd.DataFrame(results)
        self.model_objects = model_objects
        self.best_params = {"n_clusters": best_k}  # sugestão, não imposição
        self.selection_metric = selection_metric

        # O fitted_model default é o do best_k (sugestão)
        # mas o usuário pode explorar outros via summary(k=N)
        self._model = model_objects[best_k]
        self.n_components = best_k

        # ============ LRT (Nested + Bootstrapped) ============
        self.lrt_results = self._run_lrt(
            k_values=param_grid.get("n_clusters", [self.n_clusters]),
            n_bootstraps=50  # default; pode ser sobrescrito
        )

        return self._model

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

        result = compute_clustering_metrics(
            self.fitted_model,
            self.X,
            None
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "stepmix",
            "method": "Latent Class Analysis",
            "random_seed": self._random_seed,
            "N_observations": len(self.X),
            "N_variables": self.X.shape[1],
            "parameters": {
                "measurement": "bernoulli",
                "n_init": 5,
                "max_iter": 500,
                "n_clusters": self.n_clusters,
            },
        }

        if hasattr(self, 'lrt_results') and self.lrt_results:
            if self.lrt_results.get('nested') is not None:
                info["lrt_nested"] = self.lrt_results['nested'].to_dict(orient='records')
            if self.lrt_results.get('bootstrapped') is not None:
                info["lrt_bootstrapped"] = self.lrt_results['bootstrapped']

        if hasattr(self.fitted_model, 'converged_'):
            info["converged"] = bool(self.fitted_model.converged_)
        if 'aic' in result and result['aic'] is not None:
            info["aic"] = result['aic']
        if 'bic' in result and result['bic'] is not None:
            info["bic"] = result['bic']
        if 'entropy' in result and result['entropy'] is not None:
            info["entropy"] = result['entropy']

        if hasattr(self, 'best_params') and self.best_params:
            info["best_params"] = str(self.best_params)

        self.model_info = info

        return result

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

    def _run_lrt(self, k_values, n_bootstraps=50):
        """
        Run Nested LRT and Bootstrapped LRT across k values.

        Nested LRT: compares k vs k-1 using log-likelihoods.
        Bootstrapped LRT: uses StepMix's bootstrapped_likelihood_ratio_test
            (slower; only runs for the best k and best_k-1).

        Args:
            k_values: List of k tested during grid search.
            n_bootstraps: Number of bootstrap samples (default 50).

        Returns:
            Dictionary with 'nested' (DataFrame) and 'bootstrapped' (dict).
        """
        from scipy import stats
        from isaric.modelevaluation.assumptions import bootstrap_lrt_lca

        results = {
            'nested': None,
            'bootstrapped': None,
        }

        # === Nested LRT: k vs k-1 ===
        if hasattr(self, 'grid_results') and self.grid_results is not None:
            df = self.grid_results.sort_values('n_clusters').reset_index(drop=True)
            nested_rows = []
            for i in range(1, len(df)):
                k = int(df.loc[i, 'n_clusters'])
                k_prev = int(df.loc[i-1, 'n_clusters'])
                if k - k_prev != 1:
                    # Só compara se forem consecutivos
                    continue

                ll_k = float(df.loc[i, 'LL'])
                ll_prev = float(df.loc[i-1, 'LL'])

                # Graus de liberdade: diferença de parâmetros
                # N_par(k) - N_par(k-1); usamos a diferença observada
                n_samples = self.X.shape[0]
                n_features = self.X.shape[1]
                # Aproximação: n_par(k) - n_par(k-1) ≈ (n_features + 1) + k - 1
                # (pode ser ajustado)
                dof_diff = n_features + 1  # aproximação conservadora

                lr_stat = 2 * (ll_k - ll_prev)
                if lr_stat < 0:
                    lr_stat = 0.0
                p_value = float(stats.chi2.sf(lr_stat, dof_diff))

                nested_rows.append({
                    'n_clusters': k,
                    'LL_k': ll_k,
                    'LL_k_prev': ll_prev,
                    'lr_statistic': lr_stat,
                    'dof_diff': dof_diff,
                    'p_value': p_value,
                })

            if nested_rows:
                results['nested'] = pd.DataFrame(nested_rows)

        # === Bootstrapped LRT: apenas para o best_k ===
        if hasattr(self, 'best_params') and self.best_params:
            best_k = self.best_params.get('n_clusters')
            if best_k and best_k > 2:
                results['bootstrapped'] = bootstrap_lrt_lca(
                    X=self.X.values,
                    n_components=best_k,
                    n_bootstraps=n_bootstraps,
                    measurement="bernoulli",
                    random_state=self._random_seed,
                )

        return results

    # ======================================================================
    # PLOT METHODS (CALLED BY plots_map)
    # ======================================================================

    def _profile_heatmap(self, backend="plotly"):
        """Generate LCA class profiles heatmap."""
        from isaric.visualization.heatmaps import lca_profile_heatmap

        return lca_profile_heatmap(
            self.result_df,
            n_clusters=self.n_clusters,  # ← padronizado
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

    def _radar_plot(self, backend="plotly"):
        """Generate LCA radar plot."""
        from isaric.visualization.heatmaps import lca_radar_plot

        return lca_radar_plot(
            self.result_df,
            n_clusters=self.n_clusters,  # ← padronizado
            title="LCA Class Profiles (Radar)",
            backend=backend
        )

    def _information_criteria(self, backend="plotly"):
        """Generate information criteria plot from grid search results."""
        from isaric.visualization.lineplots import information_criteria_plot

        # Se não houve grid search, não há dados
        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "Information criteria plot requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        return information_criteria_plot(
            grid_results=self.grid_results,
            metrics=["AIC", "BIC", "CAIC", "SABIC"],
            x_col="n_clusters",
            title="Information Criteria by Number of Components",
            xaxis_title="Number of Components (k)",
            yaxis_title="Information Criteria Value",
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
            "information_criteria": self._information_criteria,
        }

    @classmethod
    def create(
        cls,
        data: pd.DataFrame,
        model: str = "kmeans",
        predictors: Optional[List[str]] = None,
        n_clusters: int = 3,
        random_seed: int = 42,
        **params
    ) -> "KMeans":
        """
        Configure and instantiate the KMeans pipeline.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier (must be "kmeans").
            predictors: Numeric variable names.
            n_clusters: Number of clusters.
            random_seed: Seed for reproducibility.

        Returns:
            KMeans instance ready for training.
        """
        model_config, X = create_kmeans_model(
            data=data,
            predictors=predictors,
            n_clusters=n_clusters,
            random_seed=random_seed
        )

        instance = cls(
            model=model_config,
            X=X,
            predictors=predictors,
            n_clusters=n_clusters,
            **params
        )
        instance._random_seed = random_seed
        
        return instance
    
    # ======================================================================
    # PRIVATE METHODS (CALLED BY fit() AND validation())
    # ======================================================================

    def _train_model(self, grid_search=False, param_grid=None, selection_metric="auto"):
        """
        Train KMeans model, optionally with grid search.

        Args:
            grid_search: Enable grid search over n_clusters.
            param_grid: Dict like {"n_clusters": [2, 3, 4, 5]}.
            selection_metric: "silhouette" (default), "calinski_harabasz", or "auto".

        Returns:
            Fitted KMeans model.
        """
        if grid_search:
            from isaric.modelevaluation.gridsearch import run_clustering_grid_search

            if param_grid is None:
                param_grid = {"n_clusters": list(range(2, 11))}

            if selection_metric == "auto":
                selection_metric = "silhouette"

            best_model, best_params, results_df = run_clustering_grid_search(
                estimator=self._model,
                X=self.X,
                param_grid=param_grid,
                selection_metric=selection_metric
            )

            self._model = best_model
            self.n_clusters = best_params.get("n_clusters", self.n_clusters)
            self.grid_results = results_df
            self.best_params = best_params

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

        result = compute_clustering_metrics(
            self.fitted_model,
            self.X,
            None
        )

        # ============ MODEL INFO ============
        info = {
            "model_type": self.model_type,
            "framework": "scikit-learn",
            "method": "K-Means",
            "random_seed": self._random_seed,
            "N_observations": len(self.X),
            "N_variables": self.X.shape[1],
            "parameters": {
                "n_clusters": self.n_clusters,
                "n_init": (
                    self.fitted_model.n_init
                    if hasattr(self.fitted_model, 'n_init')
                    else 10
                ),
            },
        }

        if hasattr(self, 'best_params') and self.best_params:
            info["best_params"] = str(self.best_params)

        self.model_info = info

        return result

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

    def _information_criteria(self, backend="plotly"):
        """Generate grid search results plot."""
        from isaric.visualization.lineplots import information_criteria_plot

        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "Information criteria plot requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        # KMeans usa silhouette (não informação criteria), mas podemos
        # exibir o score do grid search
        metrics = [
            c for c in self.grid_results.columns
            if c in ("silhouette", "calinski_harabasz", "score")
        ]
        if not metrics:
            raise ValueError("No score column found in grid_results.")

        return information_criteria_plot(
            grid_results=self.grid_results,
            metrics=metrics,
            x_col="n_clusters",
            title="Clustering Score by Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title=metrics[0],
            backend=backend
        )