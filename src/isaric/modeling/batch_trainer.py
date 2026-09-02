"""
Treinador em lote de modelos preditivos.

Treina vários algoritmos numa única chamada e reúne os resultados numa
tabela comparativa, eliminando o trabalho manual de instanciar cada classe,
chamar `fit` uma a uma e copiar métricas para uma planilha à parte.

**Não elege vencedor.** A tabela é explicitamente exploratória e a escolha
final é do pesquisador, registrada por `decide()`. Essa fronteira é
deliberada: o FR013 do contrato do pacote restringe a comparação automática
entre tipos diferentes de modelo para evitar viés de interpretação, e
algoritmos de ML não dispõem de um teste estatístico que sustente eleger um
vencedor — a comparação se faz por métricas, sem teste de hipótese.

Uso:

    from isaric.modeling.batch_trainer import RAPID_BatchTrainer

    lote = RAPID_BatchTrainer(
        models=["logistic_l2", "random_forest", "xgboost"],
        data=df, dependent_var="desfecho", independent_vars=[...],
        year_column="ano", train_end_year=2022, test_start_year=2023,
    )
    lote.fit()
    lote.summary()                      # tabela comparativa
    escolhido = lote.select("xgboost")  # decisão do pesquisador
"""

import warnings

import pandas as pd

from isaric.modeling.pipeline_factory import RAPID_PipelineFactory
from isaric.modeling.state import RAPID_StateMixin, requires_state

#: Algoritmos preditivos disponíveis para treino em lote.
PREDICTIVE_MODELS = ("logistic_l2", "decision_tree", "random_forest", "svm", "xgboost")

#: Métricas exibidas na tabela comparativa, na ordem.
COMPARISON_METRICS = (
    "auc_roc", "auc_pr", "f1", "precision", "recall",
    "specificity", "npv", "brier_score",
)


class RAPID_BatchTrainer(RAPID_StateMixin):
    """
    Treina uma lista de algoritmos preditivos e compara os resultados.

    Args:
        models (list, optional): Nomes registrados na `RAPID_PipelineFactory`.
            Default: todos os cinco algoritmos preditivos.
        model_params (dict, optional): Parâmetros específicos por algoritmo,
            no formato {"xgboost": {"max_depth_grid": [3]}}. Útil porque as
            grades de hiperparâmetros diferem entre algoritmos.
        **common_params: Parâmetros repassados a todos os pipelines (data,
            dependent_var, independent_vars, year_column, train_end_year,
            test_start_year, etc.).

    Attributes:
        models_ (dict): Nome do algoritmo -> instância do pipeline treinado.
        comparison_ (pandas.DataFrame): Tabela comparativa de performance.
        failures_ (dict): Nome do algoritmo -> erro, para os que falharam.
    """

    def __init__(self, models=None, model_params=None, **common_params):
        self._init_states()
        self.models = list(models) if models else list(PREDICTIVE_MODELS)
        self.model_params = model_params or {}
        self.common_params = common_params

        factory = RAPID_PipelineFactory()
        available = factory.available()
        unknown = [m for m in self.models if m not in available]
        if unknown:
            raise ValueError(
                f"Algoritmo(s) não registrado(s) na factory: {unknown}. "
                f"Disponíveis: {available}"
            )

        self._factory = factory
        self.models_ = {}
        self.comparison_ = None
        self.failures_ = {}

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True, **fit_params):
        """
        Treina os algoritmos, um a um, e monta a tabela comparativa.

        O treino é sequencial de propósito: cada pipeline já paraleliza
        internamente a busca de hiperparâmetros, e rodar os algoritmos em
        paralelo faria as duas camadas disputarem os mesmos núcleos. Além
        disso, sequencial permite mostrar progresso enquanto roda.

        Se um algoritmo falhar, os demais continuam: o erro é registrado em
        `failures_` e reportado ao final. Perder um lote inteiro porque um
        estimador não convergiu seria pior do que seguir com os que deram
        certo.

        Args:
            verbose (bool): Informa o progresso de cada algoritmo.
            **fit_params: Repassados ao `fit()` de cada pipeline.
        """
        self.models_ = {}
        self.failures_ = {}

        for position, name in enumerate(self.models, start=1):
            if verbose:
                print(f"[{position}/{len(self.models)}] treinando {name}...", flush=True)
            try:
                params = {**self.common_params, **self.model_params.get(name, {})}
                pipeline = self._factory.create(name, **params)
                pipeline.fit(**fit_params)
                self.models_[name] = pipeline
            except Exception as error:  # noqa: BLE001 — um algoritmo não pode derrubar o lote
                self.failures_[name] = f"{type(error).__name__}: {error}"
                if verbose:
                    print(f"    falhou — {type(error).__name__}: {error}", flush=True)

        if not self.models_:
            raise RuntimeError(
                "Nenhum algoritmo treinou com sucesso. Erros por algoritmo: "
                f"{self.failures_}"
            )

        self.comparison_ = self._build_comparison()
        self._mark_fitted()

        if self.failures_:
            warnings.warn(
                f"{len(self.failures_)} algoritmo(s) falharam e ficaram fora da "
                f"comparação: {sorted(self.failures_)}. Ver `failures_`.",
                RuntimeWarning, stacklevel=2,
            )
        return self

    @requires_state("is_fitted")
    def summary(self, table_format: str = "full"):
        """
        Exibe a tabela comparativa de performance.

        Args:
            table_format (str): 'full' mostra todas as métricas; 'short'
                limita às principais de discriminação.
        """
        table = self.comparison_
        if table_format == "short":
            columns = [c for c in ("model", "auc_roc", "auc_pr", "f1", "threshold")
                       if c in table.columns]
            table = table[columns]

        print("=" * 88)
        print("COMPARAÇÃO EXPLORATÓRIA ENTRE ALGORITMOS")
        print("=" * 88)
        print(table.to_string(index=False))
        print("-" * 88)
        print("Tabela exploratória: as métricas não são acompanhadas de teste")
        print("estatístico e não elegem um vencedor. A escolha do modelo é do")
        print("pesquisador — registre-a com `select(nome)` e `decide()`.")
        if self.failures_:
            print("-" * 88)
            print("Algoritmos que falharam:")
            for name, error in self.failures_.items():
                print(f"  {name}: {error}")
        print("=" * 88)

    @requires_state("is_fitted")
    def select(self, name: str):
        """
        Devolve o pipeline escolhido pelo pesquisador.

        Não há escolha automática: o nome é informado por quem analisa a
        tabela. O pipeline devolvido é o objeto treinado, pronto para
        `decide()`, `validation()`, `save()` e `report()`.

        Args:
            name (str): Nome do algoritmo, como usado no treino.

        Returns:
            O pipeline treinado correspondente.
        """
        if name not in self.models_:
            raise ValueError(
                f"'{name}' não está entre os algoritmos treinados com sucesso: "
                f"{sorted(self.models_)}."
            )
        return self.models_[name]

    @requires_state("is_fitted")
    def report(self, output_dir: str = "."):
        """
        Gera o relatório de cada algoritmo treinado e grava a tabela
        comparativa em CSV.

        Returns:
            dict: 'comparison' com o caminho do CSV e 'reports' com o
            caminho do relatório de cada algoritmo.
        """
        from pathlib import Path

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        comparison_path = directory / "comparacao_algoritmos.csv"
        self.comparison_.to_csv(comparison_path, index=False)

        reports = {
            name: pipeline.report(output_dir=directory)
            for name, pipeline in self.models_.items()
        }
        return {"comparison": str(comparison_path), "reports": reports}

    # ------------------------------------------------------------------
    # PRIVATE METHODS
    # ------------------------------------------------------------------

    def _build_comparison(self) -> pd.DataFrame:
        """Reúne as métricas já calculadas por cada pipeline, sem recalcular nada."""
        rows = []
        for name, pipeline in self.models_.items():
            metrics = pipeline.performance_metrics_ or {}
            row = {"model": name, "threshold": pipeline.threshold_}
            row.update({
                metric: metrics.get(metric)
                for metric in COMPARISON_METRICS if metric in metrics
            })
            row["best_params"] = pipeline.best_params_
            rows.append(row)

        comparison = pd.DataFrame(rows)
        numeric = [c for c in COMPARISON_METRICS if c in comparison.columns]
        comparison[numeric] = comparison[numeric].astype(float).round(6)
        return comparison
