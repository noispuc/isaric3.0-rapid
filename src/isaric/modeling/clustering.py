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
    random_seed: int = 42,
    outcome: Optional[str] = None
) -> Tuple[StepMix, pd.DataFrame, Optional[pd.Series]]:
    """
    Configure a Latent Class Analysis (LCA) model (not fitted).

    Args:
        data: Input DataFrame in ARC format.
        measurement_vars: List of binary variables.
        n_clusters: Number of latent classes (default 3).
        n_init: Number of random initializations.
        max_iter: Maximum EM iterations.
        random_seed: Seed for reproducibility.
        outcome: Optional name of an outcome column for post-hoc
            analysis (crosstab with classes).

    Returns:
        Tuple of (model, X, outcome_series):
            - model: StepMix instance.
            - X: Binary matrix.
            - outcome_series: Series with outcome values (or None).

    Raises:
        ValueError: If columns are not binary or missing.
    """
    _validate_binary_columns(data, measurement_vars)

    X = data[measurement_vars].copy()

    outcome_series = None
    if outcome is not None:
        if outcome not in data.columns:
            raise ValueError(f"Outcome column '{outcome}' not found in DataFrame.")
        if outcome in measurement_vars:
            raise ValueError(
                f"Outcome column '{outcome}' cannot also be a measurement_var."
            )
        outcome_series = data[outcome].copy()

    model = StepMix(
        n_components=n_clusters,
        measurement="bernoulli",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_seed,
        verbose=0,
    )

    return model, X, outcome_series


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
    feature_names: List[str],
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build class profiles DataFrame from fitted LCA model.

    Args:
        model: Fitted StepMix model.
        X: Binary matrix used for fitting.
        feature_names: Variable names.
        class_names: Optional display names for classes.

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

    # Index: usa class_names se fornecido, senão Class_1, Class_2, ...
    if class_names is not None and len(class_names) == n_classes:
        prof_df.index = list(class_names)
    else:
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
        class_names: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        outcome_data: Optional[pd.Series] = None,
        **kwargs
    ):
        """
        Initialize LCA with configured model and data.

        Args:
            model: Configured StepMix (from create_lca_model).
            X: Binary matrix for training.
            measurement_vars: Binary variable names.
            n_clusters: Number of latent classes.
            class_names: Optional display names for classes.
            outcome: Optional name of outcome column for crosstab.
            outcome_data: Optional Series with outcome values.
        """
        self._model = model
        self.X = X
        self.measurement_vars = measurement_vars
        self.n_clusters = n_clusters
        self.class_names = class_names
        self.outcome = outcome
        self.outcome_data = outcome_data
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
            "ll_curve": self._ll_curve,
            "entropy_curve": self._entropy_curve,
            "relative_entropy_curve": self._relative_entropy_curve,
            "bootstrap_bic_ci": self._bootstrap_bic_ci,
            "bootstrap_entropy_ci": self._bootstrap_entropy_ci,
            "crosstab": self._crosstab,
            "selection_rate": self._selection_rate,
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
        class_names: Optional[List[str]] = None,
        outcome: Optional[str] = None,
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
            class_names: Optional display names for the classes.
            outcome: Optional name of an outcome column for crosstab.

        Returns:
            LCA instance ready for training.
        """
        model_config, X, outcome_series = create_lca_model(
            data=data,
            measurement_vars=measurement_vars,
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_seed=random_seed,
            outcome=outcome
        )

        instance = cls(
            model=model_config,
            X=X,
            measurement_vars=measurement_vars,
            n_clusters=n_clusters,
            class_names=class_names,
            outcome=outcome,
            outcome_data=outcome_series,
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
            param_grid: {"n_clusters": [2, 3, 4, 5]} or None (default 2..10).
            selection_metric: "BIC", "AIC", "CAIC", "SABIC", or "auto" (=BIC).

        Returns:
            Fitted StepMix model (the default one, per selection_metric).
        """

        from scipy import stats
        
        # Auto-ativa grid_search se param_grid foi passado
        if not grid_search and param_grid is not None:
            grid_search = True

        # LCA exige grid search ou n_clusters explícito
        if not grid_search and param_grid is None:
            raise ValueError(
                "LCA requires class enumeration. "
                "Pass either `grid_search=True` or "
                "`param_grid={'n_clusters': [...]}`."
            )
        
        # ============ GRID SEARCH ============
        if param_grid is None:
            param_grid = {"n_clusters": list(range(2, 11))}

        if selection_metric == "auto":
            selection_metric = "BIC"

        n_samples = self.X.shape[0]
        n_col = self.X.shape[1]

        results = []
        model_objects = {}
        best_score = np.inf
        best_k = self.n_clusters

        # Buffers para diagnóstico (Fase 1.1)
        post_prob_dict = {}     # k → DataFrame de posteriors
        ave_pp_dict = {}        # k → list de AvePP por classe
        occ_ratios_dict = {}    # k → list de OCC por classe
        var_relevance_dict = {} # k → Series (max por variável)

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

            # Entropia relativa (Fase 1.1 - arquivo 8)
            relentropy = (
                1 - entropy / (n_samples * np.log(k))
                if k > 1 else np.nan
            )

            # Graus de liberdade (Fase 1.1 - arquivo 8)
            dof = (2 ** n_col) - ((k - 1) + n_col * 2)

            # === Diagnóstico: AvePP, OCC, min_mm_f, min_class_perc ===
            post_probs = model.predict_proba(self.X)  # (n_samples, k)
            post_probs_df = pd.DataFrame(
                post_probs,
                columns=[f"Class_{i}" for i in range(k)],
                index=self.X.index
            )
            post_prob_dict[k] = post_probs_df

            modal_classes = model.predict(self.X)

            ave_pp = []
            for i in range(k):
                # Média dos posteriors dos pacientes atribuídos à classe i
                mask = (modal_classes == i)
                if mask.sum() > 0:
                    ave_pp.append(float(post_probs[mask, i].mean()))
                else:
                    ave_pp.append(np.nan)
            ave_pp_dict[k] = ave_pp

            # OCC (Odds of Correct Classification)
            pi = model.weights_  # prior
            occ_ratios = []
            for i in range(k):
                if np.isnan(ave_pp[i]) or ave_pp[i] >= 1:
                    occ_ratios.append(np.nan)
                    continue
                odds_correct = ave_pp[i] / (1 - ave_pp[i])
                if pi[i] >= 1 or pi[i] <= 0:
                    occ_ratios.append(np.nan)
                    continue
                odds_prior = pi[i] / (1 - pi[i])
                occ_ratios.append(odds_correct / odds_prior)
            occ_ratios_dict[k] = occ_ratios

            # Variável mais relevante: max do perfil por variável
            # get_mm_df() retorna DataFrame com multi-índice (model_name, param, variable)
            try:
                mmdf = model.get_mm_df()
                # Cada coluna é uma classe; pegamos o max por variável
                # e depois o mínimo entre variáveis (variável menos relevante)
                var_max = mmdf.max(axis=1)  # max por variável (entre classes)
                var_relevance_dict[k] = var_max
                min_mm_f = float(var_max.min())
            except Exception:
                min_mm_f = np.nan

            # min_class_perc: menor % de classe
            min_class_perc = float(model.weights_.min() * 100)

            results.append({
                'n_clusters': k,
                'LL': ll,
                'AIC': aic,
                'BIC': bic,
                'CAIC': caic,
                'SABIC': sabic,
                'entropy': entropy,
                'relative_entropy': relentropy,
                'ave_pp_min': float(np.nanmin(ave_pp)) if len(ave_pp) > 0 else np.nan,
                'occ_k_min': float(np.nanmin(occ_ratios)) if len(occ_ratios) > 0 else np.nan,
                'min_mm_f': min_mm_f,
                'min_class_perc': min_class_perc,
                'n_par': int(npar),
                'dof': int(dof),
                'converged': bool(model.converged_),
            })

            model_objects[k] = model

            metric_value = {
                'BIC': bic, 'AIC': aic, 'CAIC': caic, 'SABIC': sabic
            }.get(selection_metric, bic)

            if metric_value < best_score:
                best_score = metric_value
                best_k = k

        # ============ GUARDA RESULTADOS ============
        self.grid_results = pd.DataFrame(results)
        self.model_objects = model_objects
        self.best_params = {"n_clusters": best_k}
        self.selection_metric = selection_metric

        # Diagnósticos completos (Fase 1.1)
        var_relevance_df = pd.DataFrame(var_relevance_dict).T
        var_relevance_df.index.name = "n_clusters"

        # ave_pp: DataFrame com padding (k varia por linha)
        max_k = max(len(v) for v in ave_pp_dict.values()) if ave_pp_dict else 0
        ave_pp_rows = []
        ave_pp_index = []
        for k, vals in ave_pp_dict.items():
            ave_pp_rows.append(list(vals) + [np.nan] * (max_k - len(vals)))
            ave_pp_index.append(k)
        ave_pp_df = pd.DataFrame(
            ave_pp_rows,
            index=ave_pp_index,
            columns=[f"Class_{i+1}" for i in range(max_k)]
        )
        ave_pp_df.index.name = "n_clusters"

        # occ_ratios: mesmo tratamento
        occ_rows = []
        occ_index = []
        for k, vals in occ_ratios_dict.items():
            occ_rows.append(list(vals) + [np.nan] * (max_k - len(vals)))
            occ_index.append(k)
        occ_df = pd.DataFrame(
            occ_rows,
            index=occ_index,
            columns=[f"Class_{i+1}" for i in range(max_k)]
        )
        occ_df.index.name = "n_clusters"

        self.DiagClass = {
            'posterior_prob': pd.concat(post_prob_dict, axis=1),
            'ave_pp': ave_pp_df,
            'occ_ratios': occ_df,
            'var_relevance': var_relevance_df,
        }

        # O fitted_model default é o do best_k (sugestão)
        self._model = model_objects[best_k]
        self.n_clusters = best_k  # ← CORRIGIDO

        # ============ NESTED LRT (com dof correto) ============
        nested_rows = []
        df_grid = self.grid_results.sort_values('n_clusters').reset_index(drop=True)
        for i in range(1, len(df_grid)):
            k_curr = int(df_grid.loc[i, 'n_clusters'])
            k_prev = int(df_grid.loc[i-1, 'n_clusters'])
            if k_curr - k_prev != 1:
                continue
            ll_curr = float(df_grid.loc[i, 'LL'])
            ll_prev = float(df_grid.loc[i-1, 'LL'])
            npar_curr = int(df_grid.loc[i, 'n_par'])
            npar_prev = int(df_grid.loc[i-1, 'n_par'])
            dof_diff = npar_curr - npar_prev
            lr_stat = max(0.0, 2 * (ll_curr - ll_prev))
            p_value = float(stats.chi2.sf(lr_stat, dof_diff)) if dof_diff > 0 else np.nan

            nested_rows.append({
                'n_clusters': k_curr,
                'LL_k': ll_curr,
                'LL_k_prev': ll_prev,
                'lr_statistic': lr_stat,
                'dof_diff': dof_diff,
                'p_value': p_value,
            })

        self.nested_models_lrt = pd.DataFrame(nested_rows) if nested_rows else pd.DataFrame()

        # ============ LRT bootstrapped (Fase 2 vai refinar) ============
        self.lrt_results = self._run_lrt(
            k_values=param_grid.get("n_clusters", [self.n_clusters]),
            n_bootstraps=50
        )
        # Anexa o nested LRT já calculado
        if not self.nested_models_lrt.empty:
            self.lrt_results['nested'] = self.nested_models_lrt

        return self._model

    def _build_result_df(self):
        """Build class profiles DataFrame."""
        return _build_result_df(
            model=self.fitted_model,
            X=self.X,
            feature_names=self.measurement_vars,
            class_names=self.class_names  # ← NOVO
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

        # Diagnósticos do k ativo (Fase 1.1)
        if hasattr(self, 'grid_results') and self.grid_results is not None:
            row = self.grid_results[
                self.grid_results['n_clusters'] == self.n_clusters
            ]
            if len(row) > 0:
                row = row.iloc[0]
                for col in ['relative_entropy', 'ave_pp_min',
                            'occ_k_min', 'min_mm_f', 'min_class_perc',
                            'n_par', 'dof']:
                    if col in row.index and pd.notna(row[col]):
                        info[col] = (
                            float(row[col]) if isinstance(row[col], (int, float, np.floating))
                            else row[col]
                        )

        # ============ CROSSTAB COM OUTCOME EXTERNO (Fase 3.2) ============
        if self.outcome is not None and self.outcome_data is not None:
            try:
                pred = self.fitted_model.predict(self.X)
                df_ct = pd.DataFrame({
                    'class': [f"Class_{int(c)+1}" for c in pred],
                    'outcome': self.outcome_data.values,
                })
                ct = pd.crosstab(
                    df_ct['class'], df_ct['outcome'], normalize='index'
                )

                # Teste estatístico
                from scipy import stats as sps
                ct_counts = pd.crosstab(df_ct['class'], df_ct['outcome'])
                outcome_is_categorical = (
                    not pd.api.types.is_numeric_dtype(self.outcome_data)
                    or self.outcome_data.nunique() <= 10
                )

                if outcome_is_categorical:
                    chi2, p, dof, _ = sps.chi2_contingency(ct_counts)
                    test_result = {
                        'test': 'chi-square',
                        'statistic': float(chi2),
                        'p_value': float(p),
                        'dof': int(dof),
                    }
                else:
                    groups = [
                        self.outcome_data.values[pred == c]
                        for c in sorted(set(pred))
                    ]
                    f_stat, p = sps.f_oneway(*groups)
                    test_result = {
                        'test': 'ANOVA',
                        'statistic': float(f_stat),
                        'p_value': float(p),
                    }

                info["outcome"] = {
                    "name": self.outcome,
                    "crosstab": ct.round(4).to_dict(),
                    "test": test_result,
                }
            except Exception as e:
                info["outcome_error"] = str(e)

        # ============ ADICIONA LRT E BEST_PARAMS ============
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
        Run Nested LRT, Bootstrapped LRT (BLRT), and bias-corrected CI.

        For each consecutive pair (k-1, k), computes:
        - Nested LRT (chi-square approximation)
        - BLRT (bootstrap p-value)
        - Bias-corrected bootstrap CI for each metric

        Args:
            k_values: List of k tested during grid search.
            n_bootstraps: Number of bootstrap samples (default 50).

        Returns:
            Dictionary with:
            - 'nested' (DataFrame): LRT vs k-1 (chi-square).
            - 'blrt' (DataFrame): Bootstrapped LRT p-values.
            - 'ci_bootstrap' (DataFrame): Bias-corrected CIs.
            - 'bootstrap_stats' (DataFrame): Full bootstrap stats per k.
        """
        from isaric.modelevaluation.assumptions import (
            bootstrap_stepmix,
            bootstrap_lrt_stepmix,
            bootstrap_ci_stepmix,
        )

        results = {
            'nested': None,
            'blrt': None,
            'ci_bootstrap': None,
            'bootstrap_stats': None,
        }

        # === Nested LRT (já calculado em _train_model) ===
        if hasattr(self, 'nested_models_lrt') and self.nested_models_lrt is not None:
            results['nested'] = self.nested_models_lrt

        # === Prepara dados ===
        X_arr = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        k_list = sorted(self.model_objects.keys())

        if len(k_list) < 2:
            return results

        # === Bootstrap paramétrico para cada k ===
        bootstrap_stats_by_k = {}
        blrt_rows = []
        ci_rows = []

        for k in k_list:
            model_k = self.model_objects[k]
            try:
                stats_k = bootstrap_stepmix(
                    fitted_estimator=model_k,
                    X=X_arr,
                    Y=None,
                    n_bootstrap=n_bootstraps,
                    parametric=True,
                    random_state=self._random_seed,
                )
                stats_k['n_clusters'] = k
                stats_k['rep'] = range(len(stats_k))
                bootstrap_stats_by_k[k] = stats_k
            except Exception as e:
                bootstrap_stats_by_k[k] = None

        # Concatena para guardar
        stats_concat = pd.concat(
            [s for s in bootstrap_stats_by_k.values() if s is not None],
            ignore_index=True
        ) if any(s is not None for s in bootstrap_stats_by_k.values()) else None
        results['bootstrap_stats'] = stats_concat

        # === BLRT: k vs k-1 ===
        for i in range(1, len(k_list)):
            k_curr = k_list[i]
            k_prev = k_list[i-1]
            stats_curr = bootstrap_stats_by_k.get(k_curr)
            stats_prev = bootstrap_stats_by_k.get(k_prev)
            model_curr = self.model_objects[k_curr]
            model_prev = self.model_objects[k_prev]

            if stats_curr is None or stats_prev is None:
                blrt_rows.append({
                    'n_clusters': k_curr,
                    'blrt_p_value': np.nan,
                    'n_bootstraps': n_bootstraps,
                    'error': 'bootstrap failed for k or k-1',
                })
                continue

            try:
                p_value = bootstrap_lrt_stepmix(
                    null_model=model_prev,
                    alt_model=model_curr,
                    stats_null=stats_prev,
                    stats_alt=stats_curr,
                    X=X_arr,
                    Y=None,
                    n_bootstrap=n_bootstraps,
                )
                blrt_rows.append({
                    'n_clusters': k_curr,
                    'blrt_p_value': p_value,
                    'n_bootstraps': n_bootstraps,
                })
            except Exception as e:
                blrt_rows.append({
                    'n_clusters': k_curr,
                    'blrt_p_value': np.nan,
                    'n_bootstraps': n_bootstraps,
                    'error': str(e),
                })

            # === CI bias-corrected para k_curr ===
            try:
                ci_k = bootstrap_ci_stepmix(
                    fitted_estimator=model_curr,
                    stats_bootstrap=stats_curr,
                    X=X_arr,
                )
                ci_k['n_clusters'] = k_curr
                ci_rows.append(ci_k)
            except Exception as e:
                pass

        if blrt_rows:
            results['blrt'] = pd.DataFrame(blrt_rows)
        if ci_rows:
            results['ci_bootstrap'] = pd.concat(ci_rows, ignore_index=True)

        if results.get('bootstrap_stats') is not None:
            try:
                from isaric.modelevaluation.assumptions import bootstrap_selection_rate
                sel_rate = bootstrap_selection_rate(
                    bootstrap_stats=results['bootstrap_stats'],
                    k_col='n_clusters',
                    rep_col='rep',
                )
                results['selection_rate'] = sel_rate
            except Exception as e:
                results['selection_rate'] = None

        return results

    def _bootstrap_bic_ci(self, backend="plotly"):
        """Plot BIC with bootstrap CI."""
        from isaric.visualization.lineplots import bootstrap_ci_plot

        if not hasattr(self, 'lrt_results') or not self.lrt_results:
            raise ValueError("Run fit(grid_search=True) first.")
        ci_df = self.lrt_results.get('ci_bootstrap')
        if ci_df is None or ci_df.empty:
            raise ValueError("No bootstrap CI results available.")

        return bootstrap_ci_plot(
            ci_df=ci_df,
            metric="BIC",
            x_col="n_clusters",
            backend=backend
        )

    def _bootstrap_entropy_ci(self, backend="plotly"):
        """Plot entropy with bootstrap CI."""
        from isaric.visualization.lineplots import bootstrap_ci_plot

        if not hasattr(self, 'lrt_results') or not self.lrt_results:
            raise ValueError("Run fit(grid_search=True) first.")
        ci_df = self.lrt_results.get('ci_bootstrap')
        if ci_df is None or ci_df.empty:
            raise ValueError("No bootstrap CI results available.")

        return bootstrap_ci_plot(
            ci_df=ci_df,
            metric="entropy",
            x_col="n_clusters",
            backend=backend
        )

    def _selection_rate(self, backend="plotly"):
        """Plot bootstrap selection rate per metric and k."""
        from isaric.visualization.lineplots import selection_rate_plot

        if not hasattr(self, 'lrt_results') or not self.lrt_results:
            raise ValueError("Run fit(grid_search=True) first.")

        sel_rate = self.lrt_results.get('selection_rate')
        if sel_rate is None or sel_rate.empty:
            raise ValueError("No selection rate results available.")

        return selection_rate_plot(
            selection_rate_df=sel_rate,
            x_col="n_clusters",
            title="Bootstrap Selection Rate",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Selection Rate (%)",
            backend=backend
        )

    def _crosstab(self, backend="plotly"):
        """Generate crosstab heatmap between classes and outcome."""
        from isaric.visualization.heatmaps import crosstab_heatmap

        if self.outcome is None or self.outcome_data is None:
            raise ValueError(
                "No outcome column was provided. "
                "Pass `outcome=<column_name>` in create()."
            )

        # Predict classes para o modelo ativo
        pred = self.fitted_model.predict(self.X)

        # Constrói crosstab (contagens)
        df = pd.DataFrame({
            'class': [f"Class_{int(c)+1}" for c in pred],
            'outcome': self.outcome_data.values,
        })
        ct = pd.crosstab(df['class'], df['outcome'])

        # Reordena classes para ter Class_1, Class_2, ...
        ct = ct.sort_index()

        return crosstab_heatmap(
            crosstab=ct,
            title=f"Crosstab: Classes vs {self.outcome}",
            normalize=True,
            backend=backend
        )

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
            n_clusters=self.n_clusters,
            title="LCA Class Profiles (Radar)",
            class_names=self.class_names,
            backend=backend
        )

    def _information_criteria(self, backend="plotly"):
        """Generate information criteria plot from grid search results."""
        from isaric.visualization.lineplots import grid_search_metric_plot

        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "Information criteria plot requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        return grid_search_metric_plot(
            grid_results=self.grid_results,
            metrics=["AIC", "BIC", "CAIC", "SABIC"],
            x_col="n_clusters",
            convergence_col="converged",
            title="Information Criteria by Number of Components",
            xaxis_title="Number of Components (k)",
            yaxis_title="Information Criteria Value",
            backend=backend
        )

    def _ll_curve(self, backend="plotly"):
        """Generate LL vs k curve from grid search results."""
        from isaric.visualization.lineplots import grid_search_metric_plot

        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "LL curve requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        return grid_search_metric_plot(
            grid_results=self.grid_results,
            metrics=["LL"],
            x_col="n_clusters",
            convergence_col="converged",
            title="Log-Likelihood by Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Log-Likelihood",
            backend=backend
        )

    def _entropy_curve(self, backend="plotly"):
        """Generate entropy vs k curve."""
        from isaric.visualization.lineplots import grid_search_metric_plot

        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "Entropy curve requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        return grid_search_metric_plot(
            grid_results=self.grid_results,
            metrics=["entropy"],
            x_col="n_clusters",
            convergence_col="converged",
            title="Entropy by Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Entropy",
            backend=backend
        )

    def _relative_entropy_curve(self, backend="plotly"):
        """Generate relative entropy vs k curve."""
        from isaric.visualization.lineplots import grid_search_metric_plot

        if not hasattr(self, 'grid_results') or self.grid_results is None:
            raise ValueError(
                "Relative entropy curve requires grid search. "
                "Call fit(grid_search=True) or fit(param_grid={...})."
            )

        return grid_search_metric_plot(
            grid_results=self.grid_results,
            metrics=["relative_entropy"],
            x_col="n_clusters",
            convergence_col="converged",
            title="Relative Entropy by Number of Clusters",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Relative Entropy",
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