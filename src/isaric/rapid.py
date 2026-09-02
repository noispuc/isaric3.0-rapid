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
    # ABSTRACT CONTRACT
    # ======================================================================

    @classmethod
    def create(cls, data, model, **params):
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
            raise ValueError(f"Unknown model: '{model}'")

        # Delega para a subclasse
        pipeline_cls = registry[model]
        return pipeline_cls.create(data=data, model=model, **params)

    # ======================================================================
    # CONCRETE METHODS (IMPLEMENTED - INHERITED BY SUBCLASSES)
    # ======================================================================

    def fit(
        self,
        metrics: Optional[List[str]] = None,
        cross_validation: bool = False,
        k_folds: int = 5,
        repetitions: int = 1,
        calibration: bool = False
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
            cross_validation: Enable k-fold cross-validation.
            k_folds: Number of folds.
            repetitions: Number of repetitions for repeated k-fold.
            calibration: Enable calibration curve generation.

        Returns:
            self for method chaining.

        Raises:
            ValueError: If parameters are invalid or state is incorrect.
        """
        self._check_state(self._STATE_CREATED, "fit")

        # Step 3: Modelling - Treina o modelo
        self.fitted_model = self._model.fit()

        # Step 4: Model Evaluation - Calcula métricas
        from isaric.modelevaluation.metrics import compute_classification_metrics

        if hasattr(self.fitted_model, 'predict_proba'):
            y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            y_pred = self.fitted_model.predict(self.X)

        self.metrics = compute_classification_metrics(self.y, y_pred)

        # Cross-validation
        if cross_validation:
            from isaric.modelevaluation.crossvalidation import kfold_cross_validation
            cv_results = kfold_cross_validation(
                self._model, self.X, self.y,
                n_splits=k_folds,
                scoring='roc_auc'
            )
            self.cv_metrics = cv_results

        # Calibration
        if calibration:
            from isaric.modelevaluation.calibration import compute_brier_score
            if hasattr(self.fitted_model, 'predict_proba'):
                y_prob = self.fitted_model.predict_proba(self.X)[:, 1]
                self.brier_score = compute_brier_score(self.y, y_prob)

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

        # 2. Gera plots
        if plots:
            for plot in plots:
                if plot in self.plots_map:
                    self.plots_map[plot]()
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
        from datetime import datetime

        filename = f"{self.model_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.rapid"

        metadata = {
            "model_type": self.model_type,
            "created_at": datetime.now().isoformat(),
            "rapid_version": "0.1.0",
            "key_metrics": self.metrics
        }

        with open(filename, 'wb') as f:
            json.dump(metadata, f)
            pickle.dump(self.fitted_model, f)

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
            format: List of output formats ("pdf", "png", "csv").
                If None, all formats are generated.

        Returns:
            None (generates files to disk).

        Raises:
            ValueError: If state is incorrect or format is invalid.
        """
        self._check_state(self._STATE_SUMMARIZED, "report")

        if format is None:
            format = ["pdf", "png", "csv"]

        for fmt in format:
            if fmt not in ("pdf", "png", "csv"):
                raise ValueError(f"Invalid format: {fmt}")

        # Gera CSV
        if "csv" in format:
            self.result_df.to_csv("results.csv", index=False)

        # Gera PNG
        if "png" in format:
            for plot_name, plot_func in self.plots_map.items():
                fig = plot_func()
                fig.write_image(f"{plot_name}.png")

        # Gera PDF
        if "pdf" in format:
            # Lógica de geração de PDF
            pass

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