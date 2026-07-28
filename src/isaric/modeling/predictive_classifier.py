"""
Pipeline preditivo (ML) para classificação binária com validação temporal,
seguindo o Template Method de RAPID_BasePipeline e o Factory Pattern do
pacote isaric. Consolida todos os modelos preditivos do MVP (Regressão
Logística L2, Decision Tree, Random Forest, SVM, XGBoost) em um único
arquivo, já que compartilham a mesma estrutura de pipeline.
"""

from abc import abstractmethod

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from isaric.modeling.base_model import RAPID_BasePipeline
from isaric.modelevaluation.crossvalidation import build_repeated_stratified_kfold
from isaric.modelevaluation.metrics import (
    compute_extended_classification_metrics,
    select_classification_threshold,
)
from isaric.preprocessing.collinearity import collinearity_report
from isaric.preprocessing.datasplitting import temporal_train_test_split
from isaric.preprocessing.imputation import build_imputer, drop_high_missingness_columns
from isaric.preprocessing.temporalencoding import CyclicalFeatureEncoder
from isaric.visualization.shapplots import SHAPPlots


class _AutoScalePosWeightXGBClassifier(XGBClassifier):
    """
    XGBClassifier que recalcula scale_pos_weight a partir do y recebido em
    cada chamada de fit(), garantindo que o peso de desbalanceamento seja
    sempre derivado apenas dos dados de treino do fold/refit atual (fold-safe),
    sem precisar ser passado manualmente de fora do Pipeline.
    """

    def fit(self, X, y, **kwargs):
        y_arr = np.asarray(y)
        n_pos = np.sum(y_arr == 1)
        n_neg = np.sum(y_arr == 0)
        self.scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        return super().fit(X, y, **kwargs)


class RAPID_MLBaseClassifier(RAPID_BasePipeline):
    """
    Classe base (Template Method) para pipelines preditivos de classificação
    binária com scikit-learn/XGBoost, com validação temporal.

    Args:
        data (pandas.DataFrame): Dataset de entrada.
        dependent_var (str): Coluna alvo, já codificada como 0/1.
        independent_vars (list): Preditores numéricos/binários a incluir no
            modelo (variáveis categóricas devem ser pré-codificadas, ex.:
            dummies de sexo, como no notebook de tratamento).
        year_column (str): Coluna de ano usada para o split temporal
            (ex.: 'ano_sin_pri').
        train_end_year (int): Último ano (inclusive) incluído no treino.
        test_start_year (int): Primeiro ano (inclusive) incluído no teste.
        date_column (str, optional): Coluna de data usada para o encoding
            cíclico de mês (sazonalidade).
        epiweek_column (str, optional): Coluna de semana epidemiológica no
            formato AAAASS (ex.: '201920'), usada para o encoding cíclico de
            semana.
        imputation_strategy (str): 'mice' (principal), 'median' ou 'mode'
            (análise de sensibilidade).
        max_missing_frac (float): Fração máxima de missing tolerada por
            coluna antes de descartá-la (guarda pré-MICE).
        imbalance_strategy (str or None): None (usa apenas class_weight/
            scale_pos_weight), 'smote' ou 'undersample' — aplicados somente
            dentro dos folds de treino, como análise de sensibilidade.
        cv_splits (int): Número de folds do k-fold estratificado repetido.
        cv_repeats (int): Número de repetições do k-fold.
        threshold_method (str): 'f1' ou 'youden' (ver
            select_classification_threshold).
        search_method (str): 'grid' ou 'random' para a busca de
            hiperparâmetros.
        n_iter (int): Combinações testadas quando search_method='random'.
        n_jobs (int): Paralelismo repassado à busca de hiperparâmetros.
        random_state (int): Semente para reprodutibilidade.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dependent_var: str,
        independent_vars: list,
        year_column: str,
        train_end_year: int,
        test_start_year: int,
        date_column: str = None,
        epiweek_column: str = None,
        imputation_strategy: str = "mice",
        max_missing_frac: float = 0.95,
        imbalance_strategy: str = None,
        cv_splits: int = 5,
        cv_repeats: int = 5,
        threshold_method: str = "f1",
        search_method: str = "random",
        n_iter: int = 5,
        n_jobs: int = 1,
        random_state: int = 42,
    ):
        self._run_data_validations(data, dependent_var, independent_vars, year_column)
        self.data = data.copy()
        self.dependent_var = dependent_var
        self.independent_vars = list(independent_vars)
        self.year_column = year_column
        self.train_end_year = train_end_year
        self.test_start_year = test_start_year
        self.date_column = date_column
        self.epiweek_column = epiweek_column
        self.imputation_strategy = imputation_strategy
        self.max_missing_frac = max_missing_frac
        self.imbalance_strategy = imbalance_strategy
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.threshold_method = threshold_method
        self.search_method = search_method
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.dropped_columns_ = []
        self.fitted_pipeline_ = None
        self.best_params_ = None
        self.threshold_ = None
        self.performance_metrics_ = None
        self.performance_metrics_df = None
        self.collinearity_report_ = None
        self.shap_values_ = None

        self._preprocess_data()

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def fit(self):
        """
        Executa o pipeline completo: split temporal, montagem do Pipeline
        sklearn/imblearn, tuning de hiperparâmetros com k-fold repetido
        dentro do treino, seleção de threshold fold-safe e avaliação no
        bloco de teste temporal.
        """
        self._modeling()
        self._model_evaluation()

    def summary(self, performance="all", collinearity="all", plots: list = None):
        """
        Reporta os resultados do pipeline preditivo.

        Args:
            performance: 'all' mostra as métricas de performance. None pula.
            collinearity: 'all' mostra o relatório de VIF/correlação. None pula.
            plots (list): Plots adicionais. Disponível apenas em subclasses
                com TreeSHAPMixin: ['shap_summary', 'shap_beeswarm'].
        """
        self._visualization(performance, collinearity, plots)

    def report(self):
        """Mostra todas as métricas e relatórios, sem filtros."""
        self._visualization(performance="all", collinearity="all", plots=self._default_plots)

    def validate(self, method: str = "bootstrap", n_iter: int = 100, **kwargs):
        """Delegado aos módulos de validation/ (bootstrap, external)."""
        if method == "bootstrap":
            from isaric.validation.bootstrap import bootstrap_validate
            return bootstrap_validate(self.fitted_pipeline_, self.X_test, self.y_test, n_iter=n_iter, **kwargs)
        elif method == "external":
            from isaric.validation.external import external_validate
            return external_validate(self.fitted_pipeline_, **kwargs)
        else:
            raise ValueError(f"Unknown validation method: '{method}'. Available: 'bootstrap', 'external'.")

    @property
    def _default_plots(self):
        return []

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------

    def _preprocess_data(self):
        self._data_cleaning()
        self._preprocessing()

    def _data_cleaning(self):
        self.data = self.data[self._required_columns()].dropna(
            subset=[self.dependent_var, self.year_column]
        )
        self.data, self.dropped_columns_ = drop_high_missingness_columns(
            self.data, max_missing_frac=self.max_missing_frac
        )
        still_needed = set(self.independent_vars) - set(self.data.columns)
        if still_needed:
            raise ValueError(
                f"Preditor(es) descartados por missing acima de {self.max_missing_frac:.0%}: "
                f"{sorted(still_needed)}"
            )

    def _preprocessing(self):
        # Diferente das regressões (RAPID_BaseRegression), X/y não são
        # extraídos aqui: o split temporal (feito em _modeling) precisa
        # ocorrer antes de qualquer ajuste de pré-processamento, para não
        # vazar informação do teste para o treino.
        self.data[self.dependent_var] = self.data[self.dependent_var].astype(int)

    def _modeling(self):
        df_train, df_test = temporal_train_test_split(
            self.data, self.year_column, self.train_end_year, self.test_start_year
        )

        model_input_cols = self._model_input_columns()
        self.X_train = df_train[model_input_cols]
        self.y_train = df_train[self.dependent_var]
        self.X_test = df_test[model_input_cols]
        self.y_test = df_test[self.dependent_var]

        pipeline = self._build_pipeline(model_input_cols)
        cv = build_repeated_stratified_kfold(
            n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=self.random_state
        )
        param_grid = {f"estimator__{k}": v for k, v in self._param_grid.items()}

        if param_grid:
            if self.search_method == "grid":
                search = GridSearchCV(
                    pipeline, param_grid=param_grid, cv=cv, scoring="roc_auc", n_jobs=self.n_jobs
                )
            else:
                search = RandomizedSearchCV(
                    pipeline, param_distributions=param_grid, cv=cv, scoring="roc_auc",
                    n_jobs=self.n_jobs, n_iter=self.n_iter, random_state=self.random_state,
                )
            search.fit(self.X_train, self.y_train)
            self.fitted_pipeline_ = search.best_estimator_
            self.best_params_ = search.best_params_
        else:
            pipeline.fit(self.X_train, self.y_train)
            self.fitted_pipeline_ = pipeline
            self.best_params_ = {}

        self.threshold_ = self._select_threshold()

    def _select_threshold(self):
        """
        Seleciona o threshold de decisão a partir de probabilidades
        out-of-fold no bloco de treino (fold-safe): nenhuma predição usada
        aqui provém do próprio conjunto de teste temporal.
        """
        simple_cv = build_repeated_stratified_kfold(
            n_splits=self.cv_splits, n_repeats=1, random_state=self.random_state
        )
        oof_proba = cross_val_predict(
            clone(self.fitted_pipeline_), self.X_train, self.y_train,
            cv=simple_cv, method="predict_proba", n_jobs=self.n_jobs,
        )[:, 1]
        return select_classification_threshold(self.y_train, oof_proba, method=self.threshold_method)

    def _model_evaluation(self):
        y_proba_test = self.fitted_pipeline_.predict_proba(self.X_test)[:, 1]
        y_pred_test = (y_proba_test >= self.threshold_).astype(int)

        self.performance_metrics_ = compute_extended_classification_metrics(
            self.y_test, y_pred_test, y_proba_test
        )
        self._build_performance_metrics_df()
        self._test_assumptions()

    def _test_assumptions(self):
        numeric_X_train = self.X_train[self.independent_vars].select_dtypes(include=[np.number])
        if numeric_X_train.shape[1] > 0:
            self.collinearity_report_ = collinearity_report(numeric_X_train)

    def _build_performance_metrics_df(self):
        rows = [
            {"Metric": key, "Value": f"{value:.6f}" if isinstance(value, (int, float, np.floating)) else str(value)}
            for key, value in self.performance_metrics_.items()
            if key != "confusion_matrix"
        ]
        self.performance_metrics_df = pd.DataFrame(rows)

    def _validation(self):
        pass

    def _visualization(self, performance=None, collinearity=None, plots: list = None):
        if performance is not None:
            self._report_performance()
        if collinearity is not None:
            self._report_collinearity()
        if plots:
            if "shap_summary" in plots:
                self._shap_summary_plot()
            if "shap_beeswarm" in plots:
                self._shap_beeswarm_plot()

    def _report_performance(self):
        print("=" * 80)
        print(f"PERFORMANCE METRICS (threshold={self.threshold_:.4f})")
        print("=" * 80)
        print(self.performance_metrics_df.to_string(index=False))
        cm = self.performance_metrics_["confusion_matrix"]
        print("\nConfusion Matrix:")
        print(f"True Negatives:  {cm[0, 0]:>8}    False Positives: {cm[0, 1]:>8}")
        print(f"False Negatives: {cm[1, 0]:>8}    True Positives:  {cm[1, 1]:>8}")
        print("=" * 80)

    def _report_collinearity(self):
        if self.collinearity_report_ is None:
            print("Relatório de colinearidade não disponível (nenhuma coluna numérica).")
            return
        print("=" * 80)
        print("COLLINEARITY REPORT (diagnóstico — nenhuma variável removida automaticamente)")
        print("=" * 80)
        print("\nPares com correlação acima do limiar:")
        print(self.collinearity_report_["high_correlation_pairs"].to_string(index=False))
        print("\nVIF:")
        vif_display = self.collinearity_report_["vif"]
        vif_display = vif_display[~vif_display["feature"].str.lower().isin(["intercept", "const", "constant"])]
        print(vif_display.to_string(index=False))
        print("=" * 80)

    # ------------------------------------------------------------------
    # PIPELINE ASSEMBLY HELPERS
    # ------------------------------------------------------------------

    def _required_columns(self):
        cols = list(self.independent_vars) + [self.dependent_var, self.year_column]
        if self.date_column:
            cols.append(self.date_column)
        if self.epiweek_column:
            cols.append(self.epiweek_column)
        return list(dict.fromkeys(cols))

    def _model_input_columns(self):
        cols = list(self.independent_vars)
        if self.date_column and self.date_column not in cols:
            cols.append(self.date_column)
        if self.epiweek_column and self.epiweek_column not in cols:
            cols.append(self.epiweek_column)
        return cols

    def _build_pipeline(self, model_input_cols):
        preprocessor = self._build_preprocessor(model_input_cols)
        estimator = self._build_estimator()

        steps = [("preprocessor", preprocessor)]
        sampler = self._build_sampler()
        if sampler is not None:
            steps.append(("resampler", sampler))
        steps.append(("estimator", estimator))
        return ImbPipeline(steps)

    def _build_preprocessor(self, model_input_cols):
        transformers = []
        cyclical_source_cols = set()

        if self.date_column:
            transformers.append(
                ("month_cyclical", CyclicalFeatureEncoder(period=12, source="date"), [self.date_column])
            )
            cyclical_source_cols.add(self.date_column)
        if self.epiweek_column:
            transformers.append(
                ("epiweek_cyclical", CyclicalFeatureEncoder(period=52, source="epiweek"), [self.epiweek_column])
            )
            cyclical_source_cols.add(self.epiweek_column)

        numeric_cols = [c for c in model_input_cols if c not in cyclical_source_cols]
        numeric_pipeline = SkPipeline([
            ("imputer", build_imputer(strategy=self.imputation_strategy, random_state=self.random_state)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

        return ColumnTransformer(transformers, remainder="drop")

    def _build_sampler(self):
        if self.imbalance_strategy is None:
            return None
        if self.imbalance_strategy == "smote":
            return SMOTE(random_state=self.random_state)
        if self.imbalance_strategy == "undersample":
            return RandomUnderSampler(random_state=self.random_state)
        raise ValueError(
            f"imbalance_strategy inválida: '{self.imbalance_strategy}'. Use None, 'smote' ou 'undersample'."
        )

    # ------------------------------------------------------------------
    # SHAP (implementado apenas via TreeSHAPMixin, nas subclasses de árvore)
    # ------------------------------------------------------------------

    def _compute_shap(self):
        raise NotImplementedError(
            f"{type(self).__name__} não implementa SHAP nesta rodada "
            "(MVP: apenas RAPID_RandomForest e RAPID_XGBoost)."
        )

    def _shap_summary_plot(self):
        self._compute_shap()

    def _shap_beeswarm_plot(self):
        self._compute_shap()

    # ------------------------------------------------------------------
    # ABSTRACT MEMBERS (implementados por subclasse)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def _param_grid(self):
        """Grade de hiperparâmetros do estimator (sem prefixo 'estimator__')."""
        pass

    @abstractmethod
    def _build_estimator(self):
        """
        Instancia o estimador sklearn/XGBoost já configurado para lidar com
        desbalanceamento (class_weight='balanced' ou scale_pos_weight
        recalculado por fold), sem depender de dados externos: ambos os
        mecanismos são resolvidos automaticamente a partir do y de cada
        chamada de fit(), preservando o fold-safety.
        """
        pass

    # ------------------------------------------------------------------
    # NECESSARY DATA VALIDATIONS
    # ------------------------------------------------------------------

    def _run_data_validations(self, data, dependent_var, independent_vars, year_column):
        if data is None or data.empty:
            raise ValueError("data não pode ser None ou vazio.")
        if not dependent_var or dependent_var not in data.columns:
            raise ValueError(f"dependent_var '{dependent_var}' não encontrado em data.")
        if not independent_vars:
            raise ValueError("independent_vars não pode ser vazio.")
        missing_vars = [v for v in independent_vars if v not in data.columns]
        if missing_vars:
            raise ValueError(f"Preditor(es) não encontrados em data: {missing_vars}")
        if not year_column or year_column not in data.columns:
            raise ValueError(f"year_column '{year_column}' não encontrado em data.")

        unique_y = data[dependent_var].dropna().unique()
        if len(unique_y) != 2:
            raise ValueError(f"dependent_var deve ser binário. Encontrado: {unique_y}")
        if not set(unique_y).issubset({0, 1}):
            raise ValueError(
                f"dependent_var deve estar codificado como 0/1. Encontrado: {sorted(unique_y)}"
            )


class TreeSHAPMixin:
    """
    Mixin que implementa interpretabilidade via SHAP (shap.TreeExplainer)
    para modelos baseados em árvore. Usado apenas por RAPID_RandomForest e
    RAPID_XGBoost, conforme escopo do MVP (Decision Tree e SVM/LR ficam de
    fora do SHAP nesta rodada).
    """

    def _compute_shap(self, max_samples: int = 1000):
        if self.shap_values_ is not None:
            return self.shap_values_

        preprocessor = self.fitted_pipeline_.named_steps["preprocessor"]
        estimator = self.fitted_pipeline_.named_steps["estimator"]

        X_transformed = preprocessor.transform(self.X_test)
        feature_names = self._transformed_feature_names(preprocessor)

        n_samples = X_transformed.shape[0]
        sample_idx = np.arange(n_samples)
        if n_samples > max_samples:
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(sample_idx, size=max_samples, replace=False)

        explainer = shap.TreeExplainer(estimator)
        explanation = explainer(X_transformed[sample_idx])
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            # Alguns modelos (ex.: RandomForestClassifier) retornam valores
            # SHAP por classe; mantemos apenas a classe positiva (índice 1).
            values = values[:, :, 1]

        self.shap_values_ = values
        self.shap_data_ = X_transformed[sample_idx]
        self.shap_feature_names_ = feature_names
        return self.shap_values_

    @staticmethod
    def _transformed_feature_names(preprocessor):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return None

    def _shap_summary_plot(self):
        self._compute_shap()
        return SHAPPlots.summary_plot(
            self.shap_values_, self.shap_data_, feature_names=self.shap_feature_names_,
            title=f"SHAP Summary Plot — {type(self).__name__}",
            output_path=f"shap_summary_{type(self).__name__}.png",
        )

    def _shap_beeswarm_plot(self):
        self._compute_shap()
        return SHAPPlots.beeswarm_plot(
            self.shap_values_, self.shap_data_, feature_names=self.shap_feature_names_,
            title=f"SHAP Beeswarm Plot — {type(self).__name__}",
            output_path=f"shap_beeswarm_{type(self).__name__}.png",
        )


class RAPID_LogisticL2(RAPID_MLBaseClassifier):
    """Regressão Logística com penalização L2 e class_weight='balanced'."""

    @property
    def _param_grid(self):
        return {"C": [0.01, 0.1, 1, 10]}

    def _build_estimator(self):
        return LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            max_iter=1000, random_state=self.random_state,
        )


class RAPID_DecisionTree(RAPID_MLBaseClassifier):
    """Árvore de decisão com class_weight='balanced'. Fora do escopo de SHAP no MVP."""

    @property
    def _param_grid(self):
        return {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 5, 10]}

    def _build_estimator(self):
        return DecisionTreeClassifier(class_weight="balanced", random_state=self.random_state)


class RAPID_RandomForest(TreeSHAPMixin, RAPID_MLBaseClassifier):
    """Random Forest com class_weight='balanced' e interpretabilidade via SHAP."""

    @property
    def _param_grid(self):
        return {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, None]}

    def _build_estimator(self):
        return RandomForestClassifier(class_weight="balanced", random_state=self.random_state)

    @property
    def _default_plots(self):
        return ["shap_summary", "shap_beeswarm"]


class RAPID_SVM(RAPID_MLBaseClassifier):
    """SVM (kernel RBF/linear) com probability=True e class_weight='balanced'."""

    @property
    def _param_grid(self):
        return {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}

    def _build_estimator(self):
        return SVC(probability=True, class_weight="balanced", random_state=self.random_state)


class RAPID_XGBoost(TreeSHAPMixin, RAPID_MLBaseClassifier):
    """
    XGBoost com scale_pos_weight recalculado por fold (via
    _AutoScalePosWeightXGBClassifier) e interpretabilidade via SHAP.
    """

    @property
    def _param_grid(self):
        return {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
        }

    def _build_estimator(self):
        return _AutoScalePosWeightXGBClassifier(
            eval_metric="logloss", random_state=self.random_state
        )

    @property
    def _default_plots(self):
        return ["shap_summary", "shap_beeswarm"]
