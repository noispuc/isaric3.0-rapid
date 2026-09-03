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
from typing import Dict, List, Optional
from pandas import DataFrame


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

        # Delega para a subclasse
        pipeline_cls = registry[model]
        instance = pipeline_cls.create(data=data, model=model, **params)

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
        test_size: float = 0.2
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
            test_size: Proportion for test set (default 0.2).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If parameters are invalid or state is incorrect.
        """
        self._check_state(self._STATE_CREATED, "fit")

        # Step 3: Modelling - Treina o modelo
        self.fitted_model = self._train_model()

        # Build result_df
        self.result_df = self._build_result_df()

        # Step 4: Model Evaluation - Calcula métricas
        self.metrics = self._calculate_metrics(metrics)

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

        Args:
            table_format: "full" (complete) or "short" (summarized).
            plots: List of plot names to generate.

        Returns:
            None (displays in screen).

        Raises:
            ValueError: If table_format is invalid or state is incorrect.
        """
        self._check_state(self._STATE_FITTED, "summary")

        # 1. Exibe tabela
        if table_format == "full":
            print(self.result_df.to_string())
        elif table_format == "short":
            print(self.result_df[['Variable', 'p-value']].to_string())
        else:
            raise ValueError(
                f"table_format must be 'full' or 'short'. "
                f"Received: {table_format}"
            )

        # 2. Gera e exibe plots
        if plots:
            for plot in plots:
                if plot in self.plots_map:
                    fig = self.plots_map[plot]()
                    fig.show()  # ← Adicionar esta linha
                else:
                    raise ValueError(f"Unknown plot: {plot}")

        self._transition_to(self._STATE_SUMMARIZED)

    def save(self) -> "RAPID":
        """
        Persist the trained model to disk.

        Saves JSON metadata + pickle object in a single .rapid file.

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

        filename = f"{self.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.rapid"

        metadata = {
            "model_name": self.model_type,
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
            "rapid_version": "0.1.0",
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
        bootstrap: bool = False,
        n_iterations: int = 1000,
        sensitivity: bool = False,
        subgroups: Optional[Dict] = None,
        net_benefit: bool = False
    ) -> Dict:
        """
        Validate the trained model using external data.

        This method implements Step 5 (Validation) of the RAPID
        methodology.

        Subclasses MUST have:
        - self.fitted_model (trained model)

        Args:
            external_data: Independent dataset for external validation.
            bootstrap: Enable bootstrapping.
            n_iterations: Number of bootstrap iterations.
            sensitivity: Enable sensitivity analysis.
            subgroups: Dictionary defining subgroups.
            net_benefit: Enable Decision Curve Analysis.

        Returns:
            Dictionary with validation metrics.

        Raises:
            ValueError: If state is incorrect.
        """
        self._check_state(self._STATE_SUMMARIZED, "validation")

        validation_results = {}

        # External validation
        if external_data is not None:
            from isaric.validation.external import temporal_validation
            validation_results['external'] = temporal_validation(
                self.fitted_model,
                external_data,
                dependent_var=self.dependent_var,
                independent_vars=self.independent_vars
            )

        # Bootstrap
        if bootstrap:
            from isaric.validation.bootstrap import bootstrap_metrics
            validation_results['bootstrap'] = bootstrap_metrics(
                self.fitted_model, self.X, self.y,
                n_iterations=n_iterations
            )

        # Sensitivity
        if sensitivity:
            from isaric.validation.sensitivity import alternative_missing_handling
            validation_results['sensitivity'] = alternative_missing_handling(
                self.data, self.dependent_var, self.independent_vars
            )

        # Subgroups
        if subgroups:
            from isaric.validation.subgroup import stratified_metrics
            validation_results['subgroups'] = stratified_metrics(
                self.fitted_model, self.X, self.y, subgroups
            )

        # Net benefit
        if net_benefit:
            from isaric.validation.netprofit import decision_curve_analysis
            validation_results['net_benefit'] = decision_curve_analysis(
                self.y, self.fitted_model.predict_proba(self.X)[:, 1]
            )

        return validation_results

    def report(
        self,
        format: Optional[List[str]] = None
    ) -> None:
        """
        Generate a publication report.

        Subclasses MUST have:
        - self.result_df (results DataFrame)
        - self.metrics (dict with performance metrics)
        - self.plots_map (dict: plot_name → plot_function)

        Args:
            format: List of output formats.
                Accepted Values:
                    - None (generates all formats: pdf, png, csv)
                    - ["pdf"]
                    - ["png"]
                    - ["csv"]
                    - ["pdf", "png"]
                    - ["pdf", "csv"]
                    - ["png", "csv"]
                    - ["pdf", "png", "csv"]

        Returns:
            None (generates files to disk).

        Raises:
            ValueError: If state is incorrect or format is invalid.
        """
        self._check_state(self._STATE_SUMMARIZED, "report")

        # Imports necessários
        import matplotlib.pyplot as plt

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

        # Gera CSV
        if "csv" in format:
            self.result_df.to_csv("results.csv", index=False)
            print("✅ CSV gerado: results.csv")

        # Gera PNG (via Matplotlib)
        if "png" in format:
            for plot_name, plot_func in self.plots_map.items():
                fig = plot_func(backend="matplotlib")
                fig.savefig(f"{plot_name}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✅ PNG gerado: {plot_name}.png")

        # Gera PDF consolidado (via Matplotlib)
        if "pdf" in format:
            from datetime import datetime
            from matplotlib.backends.backend_pdf import PdfPages
            
            pdf_filename = f"{self.model_type}-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
            
            with PdfPages(pdf_filename) as pdf:
                # Página 1: Metadados
                fig_meta, ax_meta = plt.subplots(figsize=(10, 6))
                ax_meta.axis('off')
                
                metadata_text = (
                    f"Model: {self.model_type}\n"
                    f"Created: {datetime.now().isoformat()}\n"
                    f"RAPID Version: 0.1.0\n\n"
                    f"Key Metrics:\n"
                )
                if self.metrics:
                    for key, value in self.metrics.items():
                        if key != 'confusion_matrix':
                            metadata_text += f"  {key}: {value}\n"
                
                ax_meta.text(0.1, 0.9, metadata_text, fontsize=12, va='top', fontfamily='monospace')
                pdf.savefig(fig_meta, bbox_inches='tight')
                plt.close(fig_meta)
                
                # Páginas seguintes: Plots
                for plot_name, plot_func in self.plots_map.items():
                    fig = plot_func(backend="matplotlib")
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