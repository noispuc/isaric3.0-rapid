"""
Pipeline preditivo (ML) para classificação binária com validação temporal,
seguindo o Template Method de RAPID_BasePipeline e o Factory Pattern do
pacote isaric. Consolida todos os modelos preditivos do MVP (Regressão
Logística L2, Decision Tree, Random Forest, SVM, XGBoost) em um único
arquivo, já que compartilham a mesma estrutura de pipeline.
"""

import json
from abc import abstractmethod
from datetime import datetime
from pathlib import Path

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
from isaric.modeling.persistence import save_model
from isaric.modeling.state import RAPID_StateMixin, requires_state
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


class RAPID_MLBaseClassifier(RAPID_StateMixin, RAPID_BasePipeline):
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
        scoring (str): Métrica de otimização do GridSearchCV/RandomizedSearchCV
            (ex.: 'roc_auc', 'f1', 'average_precision').
        shap_n_bins (int): Faixas de valor por feature contínua na agregação
            SHAP usada pelos relatórios.
        shap_min_bin_size (int): Mínimo de registros para um bin SHAP ser
            reportado. Bins menores são descartados, para que a saída seja
            de fato agregada e não informação a nível de paciente
            (NFR008; RAPID Methodology §2.2).
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
        scoring: str = "roc_auc",
        shap_n_bins: int = 10,
        shap_min_bin_size: int = 10,
    ):
        self._init_states()
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
        self.scoring = scoring
        self.shap_n_bins = shap_n_bins
        self.shap_min_bin_size = shap_min_bin_size

        self.dropped_columns_ = []
        self.fitted_pipeline_ = None
        self.best_params_ = None
        self.threshold_ = None
        self.performance_metrics_ = None
        self.performance_metrics_df = None
        self.collinearity_report_ = None
        self.shap_values_ = None
        self.shap_aggregate_ = None
        self.validation_results_ = None
        self.decision_ = None
        self.reported_metrics = None

        self._preprocess_data()

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def fit(self, metrics: list = None, validation: dict = None,
            cross_validation: bool = True, k_folds: int = None,
            repetitions: int = None):
        """
        Treina e avalia o modelo (contrato §7.6: Fit cobre os passos 3 e 4).

        Executa o pipeline completo: split temporal, montagem do Pipeline
        sklearn/imblearn, tuning de hiperparâmetros com k-fold repetido
        dentro do treino, seleção de threshold fold-safe e avaliação no
        bloco de teste temporal.

        Args:
            metrics (list, optional): Subconjunto de métricas a reportar.
                None reporta todas as calculadas.
            validation (dict, optional): Se fornecido, encadeia `validation()`
                logo após o treino, ex.: {"bootstrap": True, "n_iterations": 500}.
            cross_validation (bool): Usa k-fold repetido no tuning. False
                reduz a busca a uma única partição estratificada.
            k_folds (int, optional): Sobrescreve `cv_splits`.
            repetitions (int, optional): Sobrescreve `cv_repeats`.

        Chamar `fit` novamente é permitido — a metodologia encoraja o
        refinamento iterativo —, mas reverte `is_decided` e `is_validated`,
        que se referiam ao modelo anterior.
        """
        if k_folds is not None:
            self.cv_splits = k_folds
        if repetitions is not None:
            self.cv_repeats = repetitions
        if not cross_validation:
            self.cv_repeats = 1

        self.reported_metrics = metrics
        self.shap_values_ = None
        self.shap_aggregate_ = None
        self.validation_results_ = None
        self.decision_ = None

        self._modeling()
        self._model_evaluation()
        self._mark_fitted()

        if validation:
            self.validation(**validation)
        return self

    @requires_state("is_fitted", "is_decided", "is_validated")
    def summary(self, plots: list = None, table_format: str = "full",
                performance="all", collinearity="all"):
        """
        Exibe resultados em tabelas e figuras (contrato §7.7).

        Args:
            plots (list): Plots a gerar. Nas subclasses com TreeSHAPMixin:
                ['shap_summary', 'shap_beeswarm'].
            table_format (str): 'full' mostra todas as métricas; 'short'
                limita às principais de discriminação.
            performance: 'all' mostra as métricas. None pula.
            collinearity: 'all' mostra o relatório de VIF/correlação. None pula.

        O conteúdo muda conforme o estado: uma vez validado ou decidido, o
        resumo passa a exibir também os resultados de validação e a decisão
        registrada (Reqs 8 e 12).
        """
        self._visualization(performance, collinearity, plots, table_format=table_format)

    @requires_state("is_fitted", "is_decided", "is_validated")
    def report(self, format: str = None, output_dir: str = "."):
        """
        Gera os arquivos do relatório final (FR014, contrato §7.10).

        Produz um diretório versionado contendo um JSON reprodutível — com
        parâmetros, métricas, estados e a tabela SHAP agregada — e as figuras
        em PNG. O relatório é **imutável**: gerar de novo cria um novo
        diretório, nunca sobrescreve o anterior.

        Args:
            format (str, optional): 'json', 'png', ou None para ambos.
            output_dir (str): Diretório onde o relatório é criado.

        Returns:
            dict: Caminhos gerados, com chaves 'directory', 'json' e 'plots'.
        """
        if format not in (None, "json", "png"):
            raise ValueError(f"format deve ser 'json', 'png' ou None. Recebido: '{format}'.")

        report_dir = self._new_report_dir(output_dir)
        report_dir.mkdir(parents=True)

        produced = {"directory": str(report_dir), "json": None, "plots": [], "skipped": []}
        self._ensure_shap_aggregate()
        has_shap = self.shap_aggregate_ is not None and not self.shap_aggregate_.empty

        if format in (None, "json"):
            payload = self._report_payload()
            json_path = report_dir / "report.json"
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            produced["json"] = str(json_path)

        if format in (None, "png"):
            for plot in self._default_plots:
                # Quando a supressão por tamanho de bin não deixa nenhum grupo
                # reportável, o plot SHAP é omitido — e registrado como omitido.
                # Proteger a privacidade não pode derrubar o relatório inteiro.
                if plot == "shap_beeswarm" and not has_shap:
                    produced["skipped"].append(
                        "shap_beeswarm: nenhum bin atingiu shap_min_bin_size="
                        f"{self.shap_min_bin_size}; amostra de teste pequena demais "
                        "para uma agregação que não exponha casos individuais."
                    )
                    continue
                produced["plots"].append(self._render_plot(plot, output_dir=report_dir))

        return produced

    def _new_report_dir(self, output_dir) -> Path:
        """
        Diretório novo para cada relatório.

        Relatórios são imutáveis (FR014): nunca sobrescrevemos um existente.
        Como o carimbo de tempo tem resolução de segundos, duas chamadas
        seguidas colidiriam — nesse caso, sufixamos a versão em vez de falhar,
        porque gerar um relatório novo é justamente o comportamento previsto.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(output_dir) / f"report-{type(self).__name__}-{stamp}"
        candidate, version = base, 1
        while candidate.exists():
            version += 1
            candidate = base.with_name(f"{base.name}-v{version}")
        return candidate

    @requires_state("is_fitted")
    def validation(self, external_data=None, bootstrap: bool = False,
                   n_iterations: int = 1000, sensitivity: bool = False,
                   subgroups: dict = None, net_benefit: bool = False,
                   target: str = None):
        """
        Confirma os achados e avalia a generalização (contrato §7.9).

        Cobre as cinco técnicas previstas no contrato. Sem nenhum argumento,
        executa bootstrap — a técnica de referência da metodologia.

        Args:
            external_data (pandas.DataFrame, optional): Dataset externo.
            bootstrap (bool): Reamostragem para estimar variabilidade.
            n_iterations (int): Reamostragens do bootstrap. Default 1000,
                conforme o contrato.
            sensitivity (bool): Testa robustez a perturbação dos preditores.
            subgroups (dict, optional): {coluna: [valores]} para performance
                por subgrupo. Apenas a coluna é usada para particionar.
            net_benefit (bool): Decision Curve Analysis.
            target (str, optional): Coluna alvo do dataset externo. Assume o
                mesmo `dependent_var` quando omitido.

        Returns:
            dict: Resultados por técnica executada.
        """
        if not any([external_data is not None, bootstrap, sensitivity, subgroups, net_benefit]):
            bootstrap = True

        results = {}

        if bootstrap:
            from isaric.validation.bootstrap import bootstrap_validation
            results["bootstrap"] = bootstrap_validation(
                self.fitted_pipeline_, self.X_test, self.y_test, n_iterations=n_iterations
            )

        if external_data is not None:
            from isaric.validation.external import validate_with_external_dataset
            results["external"] = validate_with_external_dataset(
                self.fitted_pipeline_, external_data, target or self.dependent_var
            )

        if sensitivity:
            from isaric.validation.sensitivity import sensitivity_analysis
            results["sensitivity"] = sensitivity_analysis(
                self.fitted_pipeline_, self.X_test, self.y_test
            )

        if subgroups:
            from isaric.validation.subgroup import subgroup_analysis
            results["subgroup"] = {}
            for column in subgroups:
                frame = self.X_test.copy()
                frame[self.dependent_var] = self.y_test
                frame[column] = self.data.loc[frame.index, column]
                results["subgroup"][column] = subgroup_analysis(
                    self.fitted_pipeline_, frame.drop(columns=[column]).assign(**{column: frame[column]}),
                    self.dependent_var, column,
                )

        if net_benefit:
            from isaric.validation.netprofit import decision_curve_analysis
            y_proba = self.fitted_pipeline_.predict_proba(self.X_test)[:, 1]
            results["net_benefit"] = decision_curve_analysis(self.y_test, y_proba)

        self.validation_results_ = results
        self._mark_validated()
        return results

    def validate(self, method: str = "bootstrap", n_iterations: int = 1000, **kwargs):
        """
        Alias retrocompatível de `validation()`.

        Mantido porque `RAPID_BasePipeline` declara `validate` como método
        abstrato e porque o MVP expunha esta assinatura. O nome do contrato é
        `validation`; prefira-o em código novo.

        Devolve o resultado bruto da técnica pedida — e não o dicionário por
        técnica de `validation()` —, preservando o formato de retorno do MVP.
        """
        if method == "bootstrap":
            return self.validation(bootstrap=True, n_iterations=n_iterations)["bootstrap"]
        if method == "external":
            return self.validation(
                external_data=kwargs.get("external_df"), target=kwargs.get("target")
            )["external"]
        raise ValueError(
            f"Unknown validation method: '{method}'. Available: 'bootstrap', 'external'."
        )

    @requires_state("is_fitted")
    def decide(self, justification: str = None):
        """
        Registra que este modelo foi escolhido como definitivo (contrato §7.10).

        A decisão é do pesquisador: o método **não** compara modelos nem elege
        um vencedor por métrica — comparar automaticamente algoritmos de tipos
        diferentes é justamente o que o FR013 restringe. Aqui, `decide` apenas
        marca o aceite explícito, que é pré-condição de `save()`.

        Args:
            justification (str, optional): Motivo da escolha, gravado nos
                metadados e no relatório.
        """
        self.decision_ = {
            "decided_at": datetime.now().isoformat(timespec="seconds"),
            "justification": justification,
        }
        self._mark_decided()
        return self

    @requires_state("is_decided")
    def save(self, directory: str = ".", name: str = None,
             require_validation: bool = False):
        """
        Persiste o modelo treinado no formato `.rapid` (FR012, contrato §7.8).

        Exige `is_decided`: o arquivo destina-se a reuso e comparação, e não
        faz sentido persistir um modelo que o pesquisador ainda não assumiu
        como definitivo.

        A ordem `validation` antes de `decide` não é obrigatória — o contrato
        deixa essa dinâmica em aberto. Quem quiser a garantia mais estrita usa
        `require_validation=True`.

        Args:
            directory (str): Diretório de destino.
            name (str, optional): Nome do arquivo. Default:
                `MODELTYPE-YYYYMMDD-HHMMSS.rapid`.
            require_validation (bool): Exige também `is_validated`.

        Returns:
            str: Caminho do arquivo gravado.
        """
        if require_validation and not self.is_validated:
            raise RuntimeError(
                "the model was not yet validated. Please run the `validation` "
                "method before, or call save(require_validation=False)."
            )
        return save_model(self, directory=directory, name=name,
                          metrics=self.performance_metrics_)

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
                    pipeline, param_grid=param_grid, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs
                )
            else:
                search = RandomizedSearchCV(
                    pipeline, param_distributions=param_grid, cv=cv, scoring=self.scoring,
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

    def _visualization(self, performance=None, collinearity=None, plots: list = None,
                       table_format: str = "full"):
        if performance is not None:
            self._report_performance(table_format=table_format)
        if collinearity is not None and table_format != "short":
            self._report_collinearity()
        if self.validation_results_:
            self._report_validation()
        if self.decision_:
            self._report_decision()
        if plots:
            if "shap_summary" in plots:
                self._shap_summary_plot()
            if "shap_beeswarm" in plots:
                self._shap_beeswarm_plot()

    def _report_validation(self):
        print("=" * 80)
        print("VALIDATION")
        print("=" * 80)
        for technique, result in self.validation_results_.items():
            if technique == "bootstrap":
                scores = np.asarray(result, dtype=float)
                lo, hi = np.percentile(scores, [2.5, 97.5])
                print(f"bootstrap ({scores.size} reamostragens): "
                      f"média {scores.mean():.4f} | IC95% [{lo:.4f}, {hi:.4f}]")
            elif technique == "net_benefit":
                best = result.loc[result["net_benefit_model"].idxmax()]
                print(f"net_benefit (DCA): maior benefício líquido "
                      f"{best['net_benefit_model']:.4f} no threshold {best['threshold']:.2f}")
            else:
                print(f"{technique}: {result}")
        print("=" * 80)

    def _report_decision(self):
        print("=" * 80)
        print("DECISION")
        print("=" * 80)
        print(f"Modelo assumido como definitivo em {self.decision_['decided_at']}.")
        if self.decision_.get("justification"):
            print(f"Justificativa: {self.decision_['justification']}")
        print("=" * 80)

    _SHORT_METRICS = ("auc_roc", "auc_pr", "f1", "recall", "specificity", "npv")

    def _report_performance(self, table_format: str = "full"):
        table = self.performance_metrics_df
        if table_format == "short":
            table = table[table["Metric"].isin(self._SHORT_METRICS)]
        elif getattr(self, "reported_metrics", None):
            table = table[table["Metric"].isin(self.reported_metrics)]

        print("=" * 80)
        print(f"PERFORMANCE METRICS (threshold={self.threshold_:.4f})")
        print("=" * 80)
        print(table.to_string(index=False))
        if table_format == "short":
            print("=" * 80)
            return
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
    # PERSISTÊNCIA E RELATÓRIO
    # ------------------------------------------------------------------

    #: Atributos que carregam dados a nível de paciente e nunca são
    #: serializados: o `.rapid` é feito para ser arquivado e compartilhado,
    #: e a saída de um RAPID deve ser agregada (NFR008; Methodology §2.2).
    _PATIENT_LEVEL_ATTRS = (
        "data", "X_train", "X_test", "y_train", "y_test",
        "shap_data_", "shap_values_",
    )

    #: Atributos tabulares que são agregados e, por isso, podem ser
    #: persistidos: métricas por indicador, SHAP por faixa de valor,
    #: colinearidade por feature e resultados de validação.
    _AGGREGATE_ATTRS = (
        "performance_metrics_df", "shap_aggregate_",
        "collinearity_report_", "validation_results_",
    )

    def __getstate__(self):
        """
        Estado serializável do pipeline, sem dados de paciente.

        O modelo treinado, os hiperparâmetros, o threshold, as métricas e a
        tabela SHAP agregada são preservados — é o que o contrato exige para
        reuso e comparação. Os blocos de treino/teste e os valores SHAP
        individuais são descartados por desenho: um objeto carregado serve
        para inferência e auditoria, não para re-executar validações que
        dependam do dataset original.
        """
        return {k: v for k, v in self.__dict__.items() if k not in self._PATIENT_LEVEL_ATTRS}

    def _ensure_shap_aggregate(self):
        """
        Calcula a tabela SHAP agregada antes de montar o relatório, para que
        ela entre no JSON mesmo quando nenhum plot é gerado. Modelos sem
        suporte a SHAP simplesmente não têm essa seção.
        """
        try:
            self._aggregate_shap()
        except (NotImplementedError, AttributeError):
            pass

    def _report_payload(self) -> dict:
        """Conteúdo agregado e reprodutível do relatório (FR014)."""
        metrics = {
            key: (value.tolist() if hasattr(value, "tolist") else value)
            for key, value in (self.performance_metrics_ or {}).items()
        }
        payload = {
            "model_type": type(self).__name__,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "state": self.state,
            "configuration": {
                "dependent_var": self.dependent_var,
                "independent_vars": self.independent_vars,
                "year_column": self.year_column,
                "train_end_year": self.train_end_year,
                "test_start_year": self.test_start_year,
                "cv_splits": self.cv_splits,
                "cv_repeats": self.cv_repeats,
                "scoring": self.scoring,
                "threshold_method": self.threshold_method,
                "imbalance_strategy": self.imbalance_strategy,
                "random_state": self.random_state,
            },
            "best_params": self.best_params_,
            "threshold": self.threshold_,
            "metrics": metrics,
            "decision": self.decision_,
        }

        if self.shap_aggregate_ is not None and not self.shap_aggregate_.empty:
            payload["shap_aggregate"] = self.shap_aggregate_.to_dict(orient="records")

        if self.validation_results_:
            payload["validation"] = {
                technique: (
                    result.to_dict(orient="records") if isinstance(result, pd.DataFrame)
                    else (list(result) if isinstance(result, (list, np.ndarray)) else result)
                )
                for technique, result in self.validation_results_.items()
            }

        return payload

    def _render_plot(self, plot: str, output_dir) -> str:
        """Gera um plot dentro do diretório do relatório, em vez do diretório atual."""
        import os

        previous = os.getcwd()
        os.chdir(output_dir)
        try:
            if plot == "shap_summary":
                return str(Path(output_dir) / Path(self._shap_summary_plot()).name)
            if plot == "shap_beeswarm":
                return str(Path(output_dir) / Path(self._shap_beeswarm_plot()).name)
            raise ValueError(f"Plot desconhecido: '{plot}'.")
        finally:
            os.chdir(previous)

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

    def _aggregate_shap(self):
        """
        Tabela SHAP agregada por faixa de valor da feature, com supressão de
        bins pequenos. É esta tabela — e não os valores por paciente — que
        alimenta os artefatos de relatório (NFR008; RAPID Methodology §2.2).
        """
        if self.shap_aggregate_ is None:
            self._compute_shap()
            self.shap_aggregate_ = SHAPPlots.aggregate_shap(
                self.shap_values_, self.shap_data_,
                feature_names=self.shap_feature_names_,
                n_bins=self.shap_n_bins, min_bin_size=self.shap_min_bin_size,
            )
        return self.shap_aggregate_

    def _shap_beeswarm_plot(self):
        """
        Beeswarm agregado: faixas interquartis por bin de valor da feature,
        sem um ponto por paciente. Substitui o beeswarm clássico no caminho
        padrão de relatório.
        """
        return SHAPPlots.aggregated_beeswarm_plot(
            self._aggregate_shap(),
            title=f"SHAP Aggregated Beeswarm — {type(self).__name__}",
            output_path=f"shap_aggregated_beeswarm_{type(self).__name__}.png",
        )


class RAPID_LogisticL2(RAPID_MLBaseClassifier):
    """Regressão Logística com penalização L2 (defaults ajustáveis pelo usuário)."""

    def __init__(
        self,
        *,
        C_grid: list = None,
        penalty: str = "l2",
        solver: str = "lbfgs",
        class_weight="balanced",
        max_iter: int = 1000,
        **kwargs,
    ):
        # Defaults reproduzem o MVP: regularização de forte a moderada
        # (Req 27); demais argumentos liberados ao usuário (Reqs 28-30).
        self.C_grid = [0.01, 0.1, 1, 10] if C_grid is None else list(C_grid)
        self.penalty = penalty
        self.solver = solver
        self.class_weight = class_weight
        self.max_iter = max_iter
        super().__init__(**kwargs)

    @property
    def _param_grid(self):
        return {"C": self.C_grid}

    def _build_estimator(self):
        return LogisticRegression(
            penalty=self.penalty, solver=self.solver, class_weight=self.class_weight,
            max_iter=self.max_iter, random_state=self.random_state,
        )


class RAPID_DecisionTree(RAPID_MLBaseClassifier):
    """Árvore de decisão (defaults ajustáveis pelo usuário). Fora do escopo de SHAP no MVP."""

    def __init__(
        self,
        *,
        max_depth_grid: list = None,
        min_samples_split_grid: list = None,
        class_weight="balanced",
        **kwargs,
    ):
        self.max_depth_grid = [2, 3, 4, 5] if max_depth_grid is None else list(max_depth_grid)
        self.min_samples_split_grid = (
            [2, 5, 10] if min_samples_split_grid is None else list(min_samples_split_grid)
        )
        self.class_weight = class_weight
        super().__init__(**kwargs)

    @property
    def _param_grid(self):
        return {"max_depth": self.max_depth_grid, "min_samples_split": self.min_samples_split_grid}

    def _build_estimator(self):
        return DecisionTreeClassifier(class_weight=self.class_weight, random_state=self.random_state)


class RAPID_RandomForest(TreeSHAPMixin, RAPID_MLBaseClassifier):
    """Random Forest (defaults ajustáveis pelo usuário) com interpretabilidade via SHAP."""

    def __init__(
        self,
        *,
        n_estimators_grid: list = None,
        max_depth_grid: list = None,
        class_weight="balanced",
        **kwargs,
    ):
        # n_estimators/max_depth liberados (Reqs 25-26): datasets com muitas
        # variáveis ou poucas linhas podem exigir grades diferentes.
        self.n_estimators_grid = [100, 200, 300] if n_estimators_grid is None else list(n_estimators_grid)
        self.max_depth_grid = [3, 5, 7, None] if max_depth_grid is None else list(max_depth_grid)
        self.class_weight = class_weight
        super().__init__(**kwargs)

    @property
    def _param_grid(self):
        return {"n_estimators": self.n_estimators_grid, "max_depth": self.max_depth_grid}

    def _build_estimator(self):
        return RandomForestClassifier(class_weight=self.class_weight, random_state=self.random_state)

    @property
    def _default_plots(self):
        return ["shap_summary", "shap_beeswarm"]


class RAPID_SVM(RAPID_MLBaseClassifier):
    """SVM (defaults ajustáveis pelo usuário), sempre com probability=True (exigido pelo threshold fold-safe)."""

    def __init__(
        self,
        *,
        C_grid: list = None,
        kernel_grid: list = None,
        class_weight="balanced",
        **kwargs,
    ):
        self.C_grid = [0.1, 1, 10] if C_grid is None else list(C_grid)
        self.kernel_grid = ["rbf", "linear"] if kernel_grid is None else list(kernel_grid)
        self.class_weight = class_weight
        super().__init__(**kwargs)

    @property
    def _param_grid(self):
        return {"C": self.C_grid, "kernel": self.kernel_grid}

    def _build_estimator(self):
        return SVC(probability=True, class_weight=self.class_weight, random_state=self.random_state)


class RAPID_XGBoost(TreeSHAPMixin, RAPID_MLBaseClassifier):
    """
    XGBoost (defaults ajustáveis pelo usuário) com scale_pos_weight recalculado
    por fold (via _AutoScalePosWeightXGBClassifier) e interpretabilidade via SHAP.
    """

    def __init__(
        self,
        *,
        n_estimators_grid: list = None,
        max_depth_grid: list = None,
        learning_rate_grid: list = None,
        eval_metric: str = "logloss",
        **kwargs,
    ):
        self.n_estimators_grid = [50, 100, 200] if n_estimators_grid is None else list(n_estimators_grid)
        self.max_depth_grid = [3, 5, 7] if max_depth_grid is None else list(max_depth_grid)
        self.learning_rate_grid = (
            [0.01, 0.1, 0.3] if learning_rate_grid is None else list(learning_rate_grid)
        )
        self.eval_metric = eval_metric
        super().__init__(**kwargs)

    @property
    def _param_grid(self):
        return {
            "n_estimators": self.n_estimators_grid,
            "max_depth": self.max_depth_grid,
            "learning_rate": self.learning_rate_grid,
        }

    def _build_estimator(self):
        return _AutoScalePosWeightXGBClassifier(
            eval_metric=self.eval_metric, random_state=self.random_state
        )

    @property
    def _default_plots(self):
        return ["shap_summary", "shap_beeswarm"]
