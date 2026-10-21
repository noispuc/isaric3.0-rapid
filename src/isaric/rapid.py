"""
RAPID: Reusable Analytical Pipelines for Infectious Diseases.

This module provides the abstract base class that defines the contract
for all analytical pipelines in the ISARIC RAPID framework.

The contract establishes the three-phase lifecycle:
  1. create()  - Configure and instantiate the pipeline
  2. fit()     - Train the model and compute evaluation metrics
  3. summary() - Display results and visualizations

Optional methods:
  - save()       - Persist the trained model
  - validation() - Validate with external data
  - report()     - Generate publication report
  - decide()     - Select models for the report

State control ensures the correct execution order:
  created → fitted → summarized → reported

Contract Note:
    Concrete methods access standardized attributes from subclasses.
    Each subclass MUST provide:
    - self._model (configured model from create)
    - self.X, self.y (data matrices)
    - self.fitted_model (trained model)
    - self.result_df (results DataFrame)
    - self.metrics (dict with performance metrics)
    - self.plots_map (dict: plot_name → plot_function)
    - self.model_type (str)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pandas import DataFrame


def _apply_balancing(X_train, y_train, method: str, random_state: int = 42):
    """Apply undersampling or SMOTE to the training set."""
    try:
        if method == "undersample":
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(random_state=random_state)
        else:  # smote
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(random_state=random_state)
    except ImportError:
        raise ImportError(
            f"balancing='{method}' requires imbalanced-learn. "
            "Install it with: pip install imbalanced-learn"
        )
    import pandas as pd
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(y_res, name=y_train.name)
    return X_res, y_res


class RAPID(ABC):
    """
    Abstract base class for all RAPID analytical pipelines.

    This class defines the contract and provides state control.
    Subclasses implement create() (abstract) and inherit concrete
    methods (fit, summary, save, validation, report, decide).
    """

    # ======================================================================
    # STATE CONSTANTS
    # ======================================================================

    _STATE_CREATED = "created"
    _STATE_FITTED = "fitted"
    _STATE_SUMMARIZED = "summarized"
    _STATE_REPORTED = "reported"

    # ======================================================================
    # CONSTRUCTOR (INHERITED BY SUBCLASSES)
    # ======================================================================

    def __init__(self):
        """Initialize with the initial state."""
        self._state = self._STATE_CREATED
        self._libraries_used = {}
        self._fit_at = None
        self._random_seed = 42  # default; subclasses podem sobrescrever

    # ======================================================================
    # STATE CONTROL (INHERITED BY SUBCLASSES)
    # ======================================================================

    def _check_state(self, required_state: str, method_name: str) -> None:
        """
        Validate that the pipeline is in the correct state.

        Args:
            required_state: The state required to call the method.
            method_name: Name of the method being called.

        Raises:
            ValueError: If the pipeline is not in the required state.
        """
        if self._state != required_state:
            raise ValueError(
                f"Cannot call {method_name}() in state '{self._state}'. "
                f"Required state: '{required_state}'."
            )

    def _transition_to(self, new_state: str) -> None:
        """
        Transition the pipeline to a new state.

        Args:
            new_state: The new state to transition to.
        """
        self._state = new_state


    # ======================================================================
    # REGISTER LIBRARIES (INHERITED BY SUBCLASSES)
    # ======================================================================

    def _register_library(self, library_name: str) -> None:
        """
        Register a library used during the pipeline execution.
        
        Args:
            library_name: Name of the library (e.g., "statsmodels", "scikit-learn").
        """
        import importlib.metadata
        
        try:
            version = importlib.metadata.version(library_name)
            self._libraries_used[library_name] = version
        except importlib.metadata.PackageNotFoundError:
            self._libraries_used[library_name] = "not_installed"    

    # ======================================================================
    # ABSTRACT CONTRACT
    # ======================================================================

    @classmethod
    def create(
        cls,
        data: DataFrame,
        model: str,
        **params
    ) -> "RAPID":
        """
        Configure and instantiate the analytical pipeline.

        This method receives data and model configuration, validates
        the model type, and returns a configured pipeline instance
        ready for training.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier.
                Options: "logistic", "survival_cox", "survival_km", "glm",
                        "lca", "decision_tree", "random_forest", "xgboost",
                        "lightgbm", "catboost", "lasso", "ridge",
                        "elastic_net", "svm", "logistic_l2", "kmeans",
                        "descriptive".
            **params: Model-specific parameters.

        Returns:
            RAPID instance configured and ready for training.

        Raises:
            ValueError: If model type is unknown or required parameters missing.
        """
        # Imports lazy - evitam circular import
        from isaric.modeling.regression import LogisticRegression, GLM
        from isaric.modeling.survival import SurvivalCox, KaplanMeier
        from isaric.modeling.clustering import LCA, KMeans
        from isaric.modeling.descriptive import Descriptive
        from isaric.modeling.treebased import DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost
        from isaric.modeling.predictive import Lasso, Ridge, ElasticNet, SVM, LogisticL2

        # Mapeia string → classe
        registry = {
            "logistic": LogisticRegression,
            "glm": GLM,
            "survival_cox": SurvivalCox,
            "survival_km": KaplanMeier,
            "lca": LCA,
            "kmeans": KMeans,
            "descriptive": Descriptive,
            "decision_tree": DecisionTree,
            "random_forest": RandomForest,
            "xgboost": XGBoost,
            "lightgbm": LightGBM,
            "catboost": CatBoost,
            "lasso": Lasso,
            "ridge": Ridge,
            "elastic_net": ElasticNet,
            "svm": SVM,
            "logistic_l2": LogisticL2,
        }

        # Valida
        if model not in registry:
            available_models = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Available models: {available_models}"
            )

        # Extrai test_size antes de repassar para a subclasse (não é parâmetro das factories)
        import numpy as np
        from sklearn.model_selection import train_test_split as _sklearn_tts

        test_size = params.pop('test_size', 0.3)
        random_seed = params.get('random_seed', 42)

        # Delega para a subclasse
        pipeline_cls = registry[model]
        instance = pipeline_cls.create(data=data, model=model, **params)

        # Split estratificado automático para modelos de classificação sklearn
        _CLASSIFICATION_MODELS = {
            "lasso", "ridge", "elastic_net", "svm", "logistic_l2",
            "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
        }
        if model in _CLASSIFICATION_MODELS and hasattr(instance, 'X') and hasattr(instance, 'y'):
            idx = np.arange(len(instance.X))
            train_idx, test_idx = _sklearn_tts(
                idx,
                test_size=test_size,
                random_state=random_seed,
                stratify=instance.y,
            )
            instance.train_idx = train_idx
            instance.test_idx  = test_idx
            instance.X_train = instance.X.iloc[train_idx].reset_index(drop=True)
            instance.X_test  = instance.X.iloc[test_idx].reset_index(drop=True)
            instance.y_train = instance.y.iloc[train_idx].reset_index(drop=True)
            instance.y_test  = instance.y.iloc[test_idx].reset_index(drop=True)

        # Registra bibliotecas usadas baseado no modelo
        instance._register_library("pandas")
        instance._register_library("numpy")

        if model in ["logistic", "glm"]:
            instance._register_library("statsmodels")
        elif model in ["survival_cox", "survival_km"]:
            instance._register_library("lifelines")
        elif model in ["lca"]:
            instance._register_library("stepmix")
        elif model in ["kmeans", "decision_tree", "random_forest", "lasso", 
                    "ridge", "elastic_net", "svm", "logistic_l2"]:
            instance._register_library("scikit-learn")
        elif model == "xgboost":
            instance._register_library("xgboost")
        elif model == "lightgbm":
            instance._register_library("lightgbm")
        elif model == "catboost":
            instance._register_library("catboost")

        return instance


    def fit(
        self,
        metrics: Optional[List[str]] = None,
        cross_validation: bool = False,
        k_folds: int = 5,
        repetitions: int = 1,
        calibration: bool = False,
        assumptions: bool = False,
        train_test: bool = False,
        test_size: float = 0.2,
        grid_search: bool = False,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        selection_metric: str = "auto",
        bootstrap_likelihood: bool = False,
        bootstrap: bool = False,
        bootstrap_type: str = "auto",
        n_iterations: int = 1000,
        threshold: Union[float, str] = 0.5,
        balancing: Optional[str] = None,
        recalibrate: bool = False,
    ) -> "RAPID":
        """
        Train the model and compute evaluation metrics.

        This method implements Steps 3 (Modelling) and 4 (Model Evaluation)
        of the RAPID methodology.

        Subclasses MUST have:
        - self._model (configured model from create)
        - self.X, self.y (data matrices)

        Args:
            metrics: List of performance metrics (None = defaults).
                Accepted Values:
                    - Classification: ["auc", "accuracy", "precision", "recall", "f1", "log_loss", "brier_score"]
                    - Regression: ["mse", "rmse", "mae", "r2", "adjusted_r2"]
                    - Survival: ["c_index"]
                    - Information: ["aic", "bic", "entropy"]
            cross_validation: Enable k-fold cross-validation.
            k_folds: Number of folds.
            repetitions: Number of repetitions for repeated k-fold.
            calibration: Enable calibration curve generation.
            assumptions: Enable assumption checking.
            train_test: Enable train/test split validation.
            test_size: Proportion for test set (default 0.2)
            grid_search: Enable grid search over hyperparameters.
                For LCA this parameter is ignored — class enumeration
                always runs. Pass param_grid to control the k range.
            param_grid: Dictionary mapping parameter names to list of values.
                If None, uses model-specific default.
            selection_metric: Metric to optimize during grid search.
                "auto" uses model-specific default.
            bootstrap_likelihood: Enable Bootstrapped Likelihood Ratio Test
                (BLRT). Only applicable to LCA models. Computationally
                expensive — runs one bootstrap per consecutive pair (k-1, k).
                Default False.
            bootstrap: Enable bootstrap confidence intervals. When True,
                calls _run_bootstrap() after model training. Default False.
            bootstrap_type: Type of bootstrap CI to compute.
                Options: "percentile", "bias_corrected", "bca", "parametric",
                "auto" (uses model-specific default).
                - "percentile": simple quantile-based CI.
                - "bias_corrected": BC correction for skewed distributions.
                - "bca": bias-corrected and accelerated (computationally
                  expensive, standard for regression coefficients).
                - "parametric": samples from the fitted model's distribution
                  (LCA only, uses bootstrap_stepmix).
                Default "auto".
            n_iterations: Number of bootstrap samples. Only used when
                bootstrap=True. Default 1000.
            balancing: Class-balancing strategy applied to the training set
                before fitting. Options: None (no balancing), "undersample"
                (random undersampling of the majority class via imblearn),
                "smote" (SMOTE oversampling via imblearn). Only affects the
                10 sklearn classification models. Default None.
            recalibrate: If True, fit a logistic recalibration model on the
                hold-out test set predictions and add calibration slope and
                intercept to self.metrics. Default False.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If parameters are invalid or state is incorrect.
        """
        self._check_state(self._STATE_CREATED, "fit")

        from datetime import datetime

        # Store bootstrap_likelihood flag for subclasses that need it (e.g. LCA)
        self._bootstrap_likelihood = bootstrap_likelihood

        # Store bootstrap params for _run_bootstrap()
        self._bootstrap = bootstrap
        self._bootstrap_type = bootstrap_type
        self._n_iterations = n_iterations

        # Auto-ativa grid_search se param_grid foi passado
        if not grid_search and param_grid is not None:
            grid_search = True

        # Apply class-balancing to X_train/y_train before training
        _CLASSIFICATION_MODELS = {
            "lasso", "ridge", "elastic_net", "svm", "logistic_l2",
            "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost",
        }
        if balancing is not None and getattr(self, 'model_type', None) in _CLASSIFICATION_MODELS:
            if balancing not in ("undersample", "smote"):
                raise ValueError(
                    f"balancing must be None, 'undersample', or 'smote'. Got: '{balancing}'"
                )
            self.X_train, self.y_train = _apply_balancing(
                self.X_train, self.y_train, method=balancing,
                random_state=getattr(self, '_random_seed', 42),
            )
        self._balancing = balancing

        # Step 3: Modelling - Treina o modelo
        self.fitted_model = self._train_model(
            grid_search=grid_search,
            param_grid=param_grid,
            selection_metric=selection_metric
        )

        # Registra timestamp do fit
        self._fit_at = datetime.now().isoformat()

        # Build result_df
        self.result_df = self._build_result_df()

        # Armazena limiar para subclasses lerem em _calculate_metrics
        self._threshold = threshold

        # Step 4: Model Evaluation - Calcula métricas
        self.metrics = self._calculate_metrics(metrics)

        # Logistic recalibration on hold-out test set
        if recalibrate and getattr(self, 'model_type', None) in _CLASSIFICATION_MODELS:
            if hasattr(self, 'X_test') and hasattr(self, 'y_test') and hasattr(self, 'fitted_model'):
                from isaric.validation.external import recalibration as _recalibration
                y_prob_test = self.fitted_model.predict_proba(self.X_test)[:, 1]
                calib = _recalibration(self.y_test.values, y_prob_test, method="platt")
                self.metrics["calibration_slope"]     = calib["slope"]
                self.metrics["calibration_intercept"] = calib["intercept"]
        self._recalibrate = recalibrate

        # Cross-validation
        if cross_validation:
            self.cv_metrics = self._cross_validate(k_folds, repetitions)

        # Calibration
        if calibration:
            self.calibration = self._calibration_curve()

        # Assumptions
        if assumptions:
            self.assumptions = self._check_assumptions()

                # Train/Test
        if train_test:
            self.train_test = self._train_test_split(test_size)

                # Bootstrap CIs
        if bootstrap:
            self.bootstrap_results = self._validate_bootstrap(n_iterations)

        self._transition_to(self._STATE_FITTED)
        return self

    def summary(
        self,
        table_format: str = "full",
        plots: Optional[List[str]] = None
    ) -> None:
        """
        Display results in tabular and graphical format.

        This method implements Step 6 (Visualization) of the RAPID
        methodology.

        Subclasses MUST have:
        - self.result_df (results DataFrame)
        - self.plots_map (dict: plot_name → plot_function)

        Subclasses MAY have:
        - self.model_info (dict: model information and metrics)
        - self.grid_results (DataFrame: grid search results)
        - self.best_params (dict: best parameters from grid search)

        Args:
            table_format: "full" (complete) or "short" (summarized).
            plots: List of plot names to generate.

        Returns:
            None (displays in screen).

        Raises:
            ValueError: If table_format is invalid or state is incorrect.
        """
        self._check_state(self._STATE_FITTED, "summary")

        # Guarda os plots solicitados
        self._displayed_plots = plots if plots else list(self.plots_map.keys())

        # ======================================================================
        # 1. MODEL INFORMATION
        # ======================================================================
        if hasattr(self, 'model_info') and self.model_info:
            print("=" * 80)
            print("MODEL INFORMATION")
            print("=" * 80)

            # Converte dict para DataFrame para exibição uniforme
            import pandas as pd
            info_df = pd.DataFrame(
                list(self.model_info.items()),
                columns=["Key", "Value"]
            )

            # Formata valores: float → 4 casas decimais, outros → str
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            info_df["Value"] = info_df["Value"].apply(_fmt)
            print(info_df.to_string(index=False))
            print()

        # ======================================================================
        # 2. RESULTS
        # ======================================================================
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        if table_format == "full":
            print(self.result_df.to_string(index=False))
        elif table_format == "short":
            # Pega as colunas disponíveis (varia por modelo)
            display_cols = [
                c for c in ['Variable', 'Feature', 'Effect', 'OddsRatio',
                            'HazardRatio', 'Coefficient', 'Importance',
                            'p-value']
                if c in self.result_df.columns
            ]
            if display_cols:
                print(self.result_df[display_cols].to_string(index=False))
            else:
                print(self.result_df.to_string(index=False))
        else:
            raise ValueError(
                f"table_format must be 'full' or 'short'. "
                f"Received: {table_format}"
            )
        print()

        # ======================================================================
        # 3. GRID SEARCH RESULTS
        # ======================================================================
        if hasattr(self, 'grid_results') and self.grid_results is not None:
            print("=" * 80)
            print("GRID SEARCH RESULTS")
            print("=" * 80)
            print(self.grid_results.to_string(index=False))

            if hasattr(self, 'best_params') and self.best_params:
                print(f"\nBest parameters: {self.best_params}")
            print()

        # ======================================================================
        # 4. PLOTS
        # ======================================================================
        if plots:
            for plot in plots:
                if plot in self.plots_map:
                    fig = self.plots_map[plot]()
                    if isinstance(fig, list):
                        for f in fig:
                            f.show()
                    else:
                        fig.show()
                else:
                    raise ValueError(f"Unknown plot: {plot}")

        self._transition_to(self._STATE_SUMMARIZED)

    def save(self) -> "RAPID":
        """
        Persist the trained model to disk.

        Saves a ZIP file containing JSON metadata + pickle object.

        Subclasses MUST have:
        - self.model_type (str)
        - self.fitted_model (trained model)
        - self.metrics (dict with performance metrics)

        Returns:
            self for method chaining.

        Raises:
            ValueError: If state is incorrect.
        """
        self._check_state(self._STATE_SUMMARIZED, "save")

        import json
        import pickle
        import zipfile
        from datetime import datetime
        import isaric

        filename = (
            f"{self.model_type}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.rapid"
        )

        metadata = {
            "model_name": self.model_type,
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
            "rapid_version": isaric.__version__,  # ← lê versão do pacote
            "library_versions": self._libraries_used,
            "key_metrics": self.metrics
        }

        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Grava metadata.json
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))

            # Grava model.pkl
            zf.writestr('model.pkl', pickle.dumps(self.fitted_model))

        self.saved_filename = filename
        return self

    def validation(
        self,
        external_data: Optional[DataFrame] = None,
        sensitivity: bool = False,
        subgroups: Optional[Dict] = None,
        net_benefit: bool = False
    ) -> Dict:
        """
        Validate the trained model using independent external data.

        This method implements Step 5 (Validation) of the RAPID
        methodology. It is reserved for performance assessment on
        independent datasets — analogous to a TEST set in ML
        (TRAIN / VALIDATION / TEST split).

        Bootstrap confidence intervals are computed during fit()
        via bootstrap=True, not here.

        Subclasses MUST have:
        - self.fitted_model (trained model)

        Args:
            external_data: Independent dataset for external validation
                (e.g. data from another site, a held-out time period,
                or data collected subsequently). Required for external
                validation metrics to be computed.
            sensitivity: Enable sensitivity analysis.
            subgroups: Dictionary defining subgroups for stratified
                performance assessment.
            net_benefit: Enable Decision Curve Analysis (DCA).

        Returns:
            Dictionary with validation results.

        Raises:
            ValueError: If state is incorrect.
        """
        self._check_state(self._STATE_SUMMARIZED, "validation")

        validation_results = {}

        # External validation — independent dataset
        if external_data is not None:
            validation_results['external'] = self._validate_external(external_data)

        # Sensitivity
        if sensitivity:
            validation_results['sensitivity'] = self._validate_sensitivity()

        # Subgroups
        if subgroups:
            validation_results['subgroups'] = self._validate_subgroups(subgroups)

        # Net benefit
        if net_benefit:
            validation_results['net_benefit'] = self._validate_net_benefit()

        return validation_results

    def report(
        self,
        format: Optional[List[str]] = None
    ) -> None:
        """
        Generate a publication report.

        Layout (page 1):
        - Header: ISARIC platform identification
        - Pipeline identification
        - Model execution context (time, version, seed)
        - Parameters used
        - Fit metrics
        - Parameter table
        - Data size

        Pages 2+: plots and assumptions.

        Args:
            format: List of output formats.
                Accepted Values:
                    - None (generates all formats: pdf, png, csv)
                    - ["pdf"], ["png"], ["csv"]
                    - ["pdf", "png"], ["pdf", "csv"], ["png", "csv"]
                    - ["pdf", "png", "csv"]

        Returns:
            None (generates files to disk).

        Raises:
            ValueError: If state is incorrect or format is invalid.
        """
        self._check_state(self._STATE_SUMMARIZED, "report")

        import matplotlib.pyplot as plt
        import isaric
        from datetime import datetime

        # Valida formatos aceitos
        if format is None:
            format = ["pdf", "png", "csv"]

        valid_formats = ["pdf", "png", "csv"]
        for fmt in format:
            if fmt not in valid_formats:
                raise ValueError(
                    f"Invalid format: {fmt}. "
                    f"Accepted values: None, {valid_formats}"
                )

        plots_to_report = getattr(
            self, '_displayed_plots', list(self.plots_map.keys())
        )

        # ==================================================================
        # CSV - resultados + ARD (CDISC)
        # ==================================================================
        if "csv" in format:
            self.result_df.to_csv("results.csv", index=False)
            print("✅ CSV gerado: results.csv")

            # ARD (Analysis Results Dataset) - CDISC
            ard_rows = []
            for _, row in self.result_df.iterrows():
                variable = row.get('Variable', '')
                for col in self.result_df.columns:
                    if col == 'Variable':
                        continue
                    ard_rows.append({
                        'stat_name': col,
                        'stat': row[col],
                        'variable': variable,
                    })
            if ard_rows:
                import pandas as pd
                ard_df = pd.DataFrame(ard_rows)
                ard_df.to_csv("results_ard.csv", index=False)
                print("✅ ARD gerado: results_ard.csv")

        # ==================================================================
        # PNG - plots
        # ==================================================================
        if "png" in format:
            for plot_name in plots_to_report:
                if plot_name not in self.plots_map:
                    continue

                fig = self.plots_map[plot_name](backend="matplotlib")

                if isinstance(fig, list):
                    for i, f in enumerate(fig):
                        f.savefig(
                            f"{plot_name}_{i}.png",
                            dpi=300,
                            bbox_inches='tight'
                        )
                        plt.close(f)
                        print(f"✅ PNG gerado: {plot_name}_{i}.png")
                else:
                    fig.savefig(
                        f"{plot_name}.png",
                        dpi=300,
                        bbox_inches='tight'
                    )
                    plt.close(fig)
                    print(f"✅ PNG gerado: {plot_name}.png")

        # ==================================================================
        # PDF - relatório consolidado
        # ==================================================================
        if "pdf" in format:
            from matplotlib.backends.backend_pdf import PdfPages

            pdf_filename = (
                f"{self.model_type}-report-"
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
            )

            with PdfPages(pdf_filename) as pdf:
                # ==========================================================
                # Página 1 - Cabeçalho + Contexto + Parâmetros + Métricas
                # ==========================================================
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')

                lines = []
                sep = "=" * 70
                thin = "-" * 70

                # Header
                lines.append(sep)
                lines.append("ISARIC CLINICAL EPIDEMIOLOGY PLATFORM")
                lines.append("RAPID - Reusable Analytics Pipeline for Infectious Disease")
                lines.append(sep)
                lines.append("")
                lines.append(f"PIPELINE: {self.model_type}")

                # Model summary
                method = "Unknown"
                framework = "Unknown"
                if hasattr(self, 'model_info') and self.model_info:
                    method = self.model_info.get('method', 'Unknown')
                    framework = self.model_info.get('framework', 'Unknown')

                lines.append(f"MODEL SUMMARY: {method.upper()}")
                lines.append(f"Framework: {framework}")
                lines.append(sep)
                lines.append("")

                # Model execution context
                lines.append("MODEL EXECUTION CONTEXT")
                lines.append(thin)
                lines.append(f"  time_fit_was_run: {self._fit_at or 'Unknown'}")
                lines.append(f"  rapid_version: {isaric.__version__}")
                lines.append(f"  random_seed: {self._random_seed}")
                lines.append("")

                # Parameters
                if hasattr(self, 'model_info') and self.model_info.get('parameters'):
                    lines.append("PARAMETERS")
                    lines.append(thin)
                    for key, value in self.model_info['parameters'].items():
                        if isinstance(value, float):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        lines.append(f"  {key:<25} {value_str}")
                    lines.append("")

                # Fit metrics
                if self.metrics:
                    lines.append("FIT METRICS")
                    lines.append(thin)
                    for key, value in self.metrics.items():
                        if key == 'confusion_matrix':
                            continue
                        if isinstance(value, float):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        lines.append(f"  {key:<25} {value_str}")
                    lines.append("")

                # Data size
                if hasattr(self, 'model_info') and self.model_info:
                    lines.append("DATA")
                    lines.append(thin)
                    for key in ['N_observations', 'N_events', 'N_censored',
                                'N_features', 'N_variables', 'N_non_events']:
                        if key in self.model_info:
                            lines.append(
                                f"  {key:<25} {self.model_info[key]}"
                            )
                    lines.append("")

                text = "\n".join(lines)
                ax.text(
                    0.03, 0.97, text,
                    fontsize=8, va='top', ha='left',
                    fontfamily='monospace',
                    transform=ax.transAxes
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ==========================================================
                # Página 2 - Parameter table
                # ==========================================================
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')

                lines = []
                lines.append(sep)
                lines.append("PARAMETER TABLE")
                lines.append(sep)
                lines.append("")

                # Formata result_df como texto
                result_str = self.result_df.to_string(index=False)
                lines.append(result_str)

                text = "\n".join(lines)
                ax.text(
                    0.03, 0.97, text,
                    fontsize=7, va='top', ha='left',
                    fontfamily='monospace',
                    transform=ax.transAxes
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                # ==========================================================
                # Página 3 - Assumptions (se disponível)
                # ==========================================================
                if hasattr(self, 'assumptions') and self.assumptions:
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis('off')

                    lines = []
                    lines.append(sep)
                    lines.append("ASSUMPTION TESTS")
                    lines.append(sep)
                    lines.append("")

                    def _fmt_assumption(v, indent=0):
                        prefix = "  " * indent
                        if isinstance(v, dict):
                            out = []
                            for k, val in v.items():
                                if isinstance(val, (dict, list)):
                                    out.append(f"{prefix}{k}:")
                                    out.append(_fmt_assumption(val, indent + 1))
                                else:
                                    out.append(f"{prefix}{k}: {val}")
                            return "\n".join(out)
                        else:
                            return f"{prefix}{v}"

                    for key, value in self.assumptions.items():
                        lines.append(f"{key}:")
                        lines.append(_fmt_assumption(value, indent=1))
                        lines.append("")

                    text = "\n".join(lines)
                    ax.text(
                        0.03, 0.97, text,
                        fontsize=8, va='top', ha='left',
                        fontfamily='monospace',
                        transform=ax.transAxes
                    )
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

                # ==========================================================
                # Páginas 4+ - plots
                # ==========================================================
                for plot_name in plots_to_report:
                    if plot_name not in self.plots_map:
                        continue

                    fig = self.plots_map[plot_name](backend="matplotlib")

                    if isinstance(fig, list):
                        for i, f in enumerate(fig):
                            pdf.savefig(f, bbox_inches='tight')
                            plt.close(f)
                            print(f"✅ Plot adicionado ao PDF: {plot_name}_{i}")
                    else:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)
                        print(f"✅ Plot adicionado ao PDF: {plot_name}")

            print(f"✅ PDF consolidado: {pdf_filename}")

        self._transition_to(self._STATE_REPORTED)

    def decide(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> None:
        """
        Select which trained models to include in the report.

        Args:
            include: List of model names TO include.
            exclude: List of model names TO exclude.

        Returns:
            None.

        Raises:
            ValueError: If state is incorrect or parameters are invalid.
        """
        self._check_state(self._STATE_REPORTED, "decide")

        # Lista modelos disponíveis
        import os
        available_models = [
            f for f in os.listdir('.') if f.endswith('.rapid')
        ]

        # Seleciona modelos
        if include is not None:
            selected = [m for m in available_models if m in include]
        elif exclude is not None:
            selected = [m for m in available_models if m not in exclude]
        else:
            selected = available_models

        self.selected_models = selected
        print(f"Selected models: {selected}")