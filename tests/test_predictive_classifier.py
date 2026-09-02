import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isaric.modeling.predictive_classifier import (
    RAPID_DecisionTree,
    RAPID_LogisticL2,
    RAPID_RandomForest,
    RAPID_SVM,
    RAPID_XGBoost,
)
from isaric.modeling.pipeline_factory import RAPID_PipelineFactory
from isaric.modeling.persistence import (
    RAPID_Decide,
    default_filename,
    load_model,
    read_metadata,
)
from isaric.modeling.state import RAPIDStateError
from isaric.validation.netprofit import decision_curve_analysis
from isaric.visualization.shapplots import SHAPPlots

TRAIN_END_YEAR = 2019
TEST_START_YEAR = 2020


def _make_df(n=160, seed=0, test_signal=0.0):
    """
    Dataset sintético com split temporal: 2018-2019 no treino, 2020 no teste.

    test_signal desloca apenas as linhas do bloco de teste, permitindo
    verificar que nada ajustado no treino depende do conteúdo do teste.
    """
    rng = np.random.default_rng(seed)
    year = np.array([2018, 2019, 2020])[np.arange(n) % 3]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    # O desfecho é gerado antes do deslocamento, para que test_signal altere
    # apenas os preditores do bloco de teste sem colapsar as classes nele.
    logit = 0.9 * x1 - 0.6 * x2
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    x1 = np.where(year >= TEST_START_YEAR, x1 + test_signal, x1)

    return pd.DataFrame({"x1": x1, "x2": x2, "ano": year, "desfecho": y})


def _base_kwargs(df=None):
    return {
        "data": _make_df() if df is None else df,
        "dependent_var": "desfecho",
        "independent_vars": ["x1", "x2"],
        "year_column": "ano",
        "train_end_year": TRAIN_END_YEAR,
        "test_start_year": TEST_START_YEAR,
        "cv_splits": 2,
        "cv_repeats": 1,
        "search_method": "grid",
        "n_jobs": 1,
    }


def _tiny_model(cls, **overrides):
    """Instancia cada algoritmo com a menor grade possível, para os testes correrem rápido."""
    grids = {
        RAPID_LogisticL2: {"C_grid": [1.0]},
        RAPID_DecisionTree: {"max_depth_grid": [3], "min_samples_split_grid": [2]},
        RAPID_RandomForest: {"n_estimators_grid": [10], "max_depth_grid": [3]},
        RAPID_SVM: {"C_grid": [1.0], "kernel_grid": ["linear"]},
        RAPID_XGBoost: {
            "n_estimators_grid": [10],
            "max_depth_grid": [2],
            "learning_rate_grid": [0.3],
        },
    }
    kwargs = {**_base_kwargs(), **grids[cls], **overrides}
    return cls(**kwargs)


ALL_MODELS = [
    RAPID_LogisticL2,
    RAPID_DecisionTree,
    RAPID_RandomForest,
    RAPID_SVM,
    RAPID_XGBoost,
]


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_MODELS)
def test_fit_produces_metrics_and_threshold(cls):
    model = _tiny_model(cls)
    model.fit()

    assert model.fitted_pipeline_ is not None
    assert 0.0 <= model.threshold_ <= 1.0
    assert model.performance_metrics_df is not None
    assert not model.performance_metrics_df.empty
    for metric in ("auc_roc", "auc_pr", "npv", "specificity", "brier_score"):
        assert metric in model.performance_metrics_


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_summary_runs_after_fit(cls):
    model = _tiny_model(cls)
    model.fit()
    model.summary(performance="all", collinearity="all")


def test_factory_creates_every_registered_predictive_pipeline():
    factory = RAPID_PipelineFactory()
    for name in ("logistic_l2", "decision_tree", "random_forest", "svm", "xgboost"):
        assert name in factory.available()


# ----------------------------------------------------------------------
# Ausência de vazamento do bloco de teste
# ----------------------------------------------------------------------

def test_train_and_test_blocks_respect_temporal_split():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    train_years = model.data.loc[model.X_train.index, "ano"]
    test_years = model.data.loc[model.X_test.index, "ano"]

    assert train_years.max() <= TRAIN_END_YEAR
    assert test_years.min() >= TEST_START_YEAR
    assert set(model.X_train.index).isdisjoint(set(model.X_test.index))


def test_scaler_is_fitted_on_training_block_only():
    """A média do StandardScaler deve reproduzir a do treino, não a da base inteira."""
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    scaler = (
        model.fitted_pipeline_
        .named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )

    np.testing.assert_allclose(
        scaler.mean_, model.X_train[["x1", "x2"]].to_numpy().mean(axis=0), rtol=1e-9
    )


def test_nothing_fitted_on_train_changes_when_test_block_changes():
    """
    Mesmo bloco de treino, bloco de teste deslocado: threshold, hiperparâmetros
    e pré-processamento têm de sair idênticos. Se qualquer um mudar, houve
    vazamento do teste para o ajuste.
    """
    baseline = RAPID_LogisticL2(**{**_base_kwargs(_make_df(test_signal=0.0)), "C_grid": [0.1, 1.0]})
    shifted = RAPID_LogisticL2(**{**_base_kwargs(_make_df(test_signal=5.0)), "C_grid": [0.1, 1.0]})
    baseline.fit()
    shifted.fit()

    assert baseline.threshold_ == shifted.threshold_
    assert baseline.best_params_ == shifted.best_params_

    def _scaler_mean(m):
        return (
            m.fitted_pipeline_.named_steps["preprocessor"]
            .named_transformers_["numeric"].named_steps["scaler"].mean_
        )

    np.testing.assert_allclose(_scaler_mean(baseline), _scaler_mean(shifted), rtol=1e-9)


# ----------------------------------------------------------------------
# Parâmetros configuráveis (Reqs 24-30)
# ----------------------------------------------------------------------

def test_defaults_reproduce_the_mvp_grids():
    """Retrocompatibilidade: sem argumentos novos, as grades são as do MVP."""
    kwargs = _base_kwargs()
    assert RAPID_LogisticL2(**kwargs)._param_grid == {"C": [0.01, 0.1, 1, 10]}
    assert RAPID_RandomForest(**kwargs)._param_grid == {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, None],
    }
    assert RAPID_DecisionTree(**kwargs)._param_grid == {
        "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 5, 10],
    }
    assert RAPID_SVM(**kwargs)._param_grid == {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
    assert RAPID_XGBoost(**kwargs)._param_grid == {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.3],
    }


def test_default_scoring_is_roc_auc_and_is_overridable():
    kwargs = _base_kwargs()
    assert RAPID_LogisticL2(**kwargs).scoring == "roc_auc"
    assert RAPID_LogisticL2(**{**kwargs, "scoring": "average_precision"}).scoring == "average_precision"


def test_custom_grid_restricts_the_search():
    model = _tiny_model(RAPID_LogisticL2, C_grid=[0.25, 0.75])
    model.fit()

    assert model.best_params_["estimator__C"] in (0.25, 0.75)


def test_logistic_estimator_parameters_are_applied():
    model = _tiny_model(
        RAPID_LogisticL2,
        penalty="l1",
        solver="liblinear",
        class_weight=None,
        max_iter=250,
    )
    estimator = model._build_estimator()

    assert estimator.penalty == "l1"
    assert estimator.solver == "liblinear"
    assert estimator.class_weight is None
    assert estimator.max_iter == 250


def test_tree_and_svm_estimator_parameters_are_applied():
    assert _tiny_model(RAPID_DecisionTree, class_weight=None)._build_estimator().class_weight is None
    assert _tiny_model(RAPID_RandomForest, class_weight=None)._build_estimator().class_weight is None

    svm = _tiny_model(RAPID_SVM, class_weight=None)._build_estimator()
    assert svm.class_weight is None
    # probability=True é pré-condição técnica do threshold fold-safe, não preferência.
    assert svm.probability is True

    assert _tiny_model(RAPID_XGBoost, eval_metric="auc")._build_estimator().eval_metric == "auc"


# ----------------------------------------------------------------------
# validate() — regressão dos imports quebrados
# ----------------------------------------------------------------------

def test_validate_bootstrap_returns_one_score_per_iteration():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    scores = model.validate(method="bootstrap", n_iterations=5)

    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_validate_external_returns_accuracy():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    external = _make_df(n=60, seed=7)[["x1", "x2", "desfecho"]]
    result = model.validate(method="external", external_df=external, target="desfecho")

    assert "accuracy" in result


def test_validate_rejects_unknown_method():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    with pytest.raises(ValueError):
        model.validate(method="nao_existe")


# ----------------------------------------------------------------------
# Validações de entrada
# ----------------------------------------------------------------------

def test_rejects_non_binary_outcome():
    df = _make_df()
    df.loc[df.index[:10], "desfecho"] = 2

    with pytest.raises(ValueError):
        RAPID_LogisticL2(**_base_kwargs(df))


def test_rejects_outcome_not_encoded_as_zero_one():
    df = _make_df()
    df["desfecho"] = df["desfecho"].map({0: "nao", 1: "sim"})

    with pytest.raises(ValueError):
        RAPID_LogisticL2(**_base_kwargs(df))


def test_rejects_unknown_predictor():
    kwargs = _base_kwargs()
    kwargs["independent_vars"] = ["x1", "coluna_inexistente"]

    with pytest.raises(ValueError):
        RAPID_LogisticL2(**kwargs)


def test_rejects_missing_year_column():
    kwargs = _base_kwargs()
    kwargs["year_column"] = "ano_que_nao_existe"

    with pytest.raises(ValueError):
        RAPID_LogisticL2(**kwargs)


def test_rejects_invalid_imbalance_strategy():
    model = _tiny_model(RAPID_LogisticL2, imbalance_strategy="estrategia_invalida")

    with pytest.raises(ValueError):
        model.fit()


# ----------------------------------------------------------------------
# SHAP agregado (N1) — saída sem informação a nível de paciente
# ----------------------------------------------------------------------

def _shap_inputs(n=400, n_features=3, seed=1):
    """SHAP values sintéticos com uma feature contínua monotônica e uma binária."""
    rng = np.random.default_rng(seed)
    continua = rng.normal(size=n)
    binaria = rng.integers(0, 2, size=n).astype(float)
    ruido = rng.normal(size=n)
    features = np.column_stack([continua, binaria, ruido])[:, :n_features]

    shap_values = np.column_stack([
        2.0 * continua + rng.normal(scale=0.05, size=n),   # efeito monotônico
        1.5 * binaria - 0.75,                              # efeito de degrau
        rng.normal(scale=0.01, size=n),                    # sem efeito
    ])[:, :n_features]

    return shap_values, features, ["continua", "binaria", "ruido"][:n_features]


def test_aggregate_shap_returns_bins_not_patients():
    shap_values, features, names = _shap_inputs(n=400)

    agg = SHAPPlots.aggregate_shap(shap_values, features, feature_names=names)

    assert not agg.empty
    assert len(agg) < 400, "a tabela agregada não pode ter uma linha por registro"
    # Cada feature particiona a amostra: a soma dos bins não excede n por feature.
    assert (agg.groupby("feature")["n"].sum() <= 400).all()
    assert {"feature", "bin_label", "n", "shap_median", "shap_p25", "shap_p75"} <= set(agg.columns)


def test_aggregate_shap_suppresses_bins_below_min_size():
    """Sem supressão, um bin com poucos pacientes ainda seria dado individual."""
    shap_values, features, names = _shap_inputs(n=400)

    agg = SHAPPlots.aggregate_shap(
        shap_values, features, feature_names=names, min_bin_size=25
    )

    assert not agg.empty
    assert agg["n"].min() >= 25


def test_aggregate_shap_suppresses_everything_when_min_size_exceeds_sample():
    shap_values, features, names = _shap_inputs(n=60)

    agg = SHAPPlots.aggregate_shap(
        shap_values, features, feature_names=names, min_bin_size=500
    )

    assert agg.empty


def test_aggregate_shap_groups_binary_feature_by_value():
    shap_values, features, names = _shap_inputs(n=400, n_features=2)

    agg = SHAPPlots.aggregate_shap(shap_values, features, feature_names=names)
    binaria = agg[agg["feature"] == "binaria"]

    assert len(binaria) == 2, "feature binária deve render exatamente dois bins"


def test_aggregate_shap_preserves_effect_direction():
    """O que a opção A perderia: direção e monotonicidade do efeito."""
    shap_values, features, names = _shap_inputs(n=400)

    agg = SHAPPlots.aggregate_shap(shap_values, features, feature_names=names)
    continua = agg[agg["feature"] == "continua"].sort_values("bin_index")

    assert continua["shap_median"].iloc[0] < 0 < continua["shap_median"].iloc[-1]
    assert continua["shap_median"].is_monotonic_increasing


def test_aggregate_shap_rejects_mismatched_shapes():
    shap_values, features, names = _shap_inputs(n=100)

    with pytest.raises(ValueError):
        SHAPPlots.aggregate_shap(shap_values[:50], features, feature_names=names)


def test_aggregated_beeswarm_plot_rejects_empty_aggregate():
    with pytest.raises(ValueError):
        SHAPPlots.aggregated_beeswarm_plot(pd.DataFrame())


@pytest.mark.parametrize("cls", [RAPID_RandomForest, RAPID_XGBoost])
def test_report_path_uses_aggregated_shap(cls, tmp_path, monkeypatch):
    """
    O caminho padrão de relatório não pode produzir plot a nível de paciente.
    Antes desta mudança, _default_plots levava o beeswarm clássico ao report().
    """
    monkeypatch.chdir(tmp_path)
    df = _make_df(n=400, seed=5)
    model = cls(**{
        **_base_kwargs(df),
        **({"n_estimators_grid": [10], "max_depth_grid": [3]} if cls is RAPID_RandomForest
           else {"n_estimators_grid": [10], "max_depth_grid": [2], "learning_rate_grid": [0.3]}),
        "shap_min_bin_size": 5,
    })
    model.fit()

    assert model._default_plots == ["shap_summary", "shap_beeswarm"]
    path = model._shap_beeswarm_plot()

    assert "aggregated" in path
    assert (tmp_path / path).exists()
    assert model.shap_aggregate_ is not None
    assert len(model.shap_aggregate_) < len(model.X_test)


# ----------------------------------------------------------------------
# Máquina de estados (Reqs 3-18)
# ----------------------------------------------------------------------

def test_states_start_false():
    model = _tiny_model(RAPID_LogisticL2)

    assert model.state == {"is_fitted": False, "is_decided": False, "is_validated": False}


@pytest.mark.parametrize("method,args", [
    ("summary", ()),
    ("report", ()),
    ("decide", ()),
    ("validation", ()),
])
def test_methods_blocked_before_fit(method, args):
    model = _tiny_model(RAPID_LogisticL2)

    with pytest.raises(RAPIDStateError):
        getattr(model, method)(*args)


def test_fit_sets_is_fitted():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    assert model.is_fitted
    assert not model.is_decided
    assert not model.is_validated


def test_states_are_read_only():
    model = _tiny_model(RAPID_LogisticL2)

    with pytest.raises(AttributeError):
        model.is_fitted = True


def test_decide_requires_fit_then_sets_is_decided():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide(justification="melhor recall no bloco temporal")

    assert model.is_decided
    assert model.decision_["justification"] == "melhor recall no bloco temporal"


def test_validation_sets_is_validated():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.validation(bootstrap=True, n_iterations=5)

    assert model.is_validated


def test_refit_resets_downstream_states():
    """A metodologia encoraja refinamento iterativo, mas a decisão anterior
    era sobre outro modelo e não pode sobreviver ao refit."""
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.validation(bootstrap=True, n_iterations=5)
    model.decide()
    assert model.is_decided and model.is_validated

    model.fit()

    assert model.is_fitted
    assert not model.is_decided
    assert not model.is_validated


# ----------------------------------------------------------------------
# save() e formato .rapid (FR012)
# ----------------------------------------------------------------------

def test_save_blocked_until_decide(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    with pytest.raises(RAPIDStateError, match="decide"):
        model.save(directory=tmp_path)


def test_save_with_require_validation_blocks_until_validated(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()

    with pytest.raises(RuntimeError, match="validat"):
        model.save(directory=tmp_path, require_validation=True)

    model.validation(bootstrap=True, n_iterations=5)
    assert model.save(directory=tmp_path, require_validation=True)


def test_saved_file_follows_naming_convention(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()

    path = Path(model.save(directory=tmp_path))

    assert path.suffix == ".rapid"
    assert path.stem.startswith("RAPID_LogisticL2-")
    assert path.exists()


def test_saved_model_carries_no_patient_data(tmp_path):
    """NFR008: o .rapid é feito para ser arquivado e compartilhado."""
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()

    loaded, _ = load_model(model.save(directory=tmp_path))

    for attribute in ("data", "X_train", "X_test", "y_train", "y_test", "shap_data_"):
        assert not hasattr(loaded, attribute), f"{attribute} vazou para o arquivo salvo"
    assert loaded.fitted_pipeline_ is not None
    assert loaded.state["is_decided"] is True


def test_save_refuses_undeclared_patient_frame(tmp_path):
    """Qualquer DataFrame novo não declarado como agregado trava a gravação."""
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()
    model.tabela_por_paciente = pd.DataFrame({"paciente": [1, 2], "valor": [3, 4]})

    with pytest.raises(ValueError, match="tabela_por_paciente"):
        model.save(directory=tmp_path)


def test_metadata_records_versions_and_metrics(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()

    metadata = read_metadata(model.save(directory=tmp_path))

    assert metadata.model_type == "RAPID_LogisticL2"
    assert metadata.created_at
    assert "scikit-learn" in metadata.library_versions
    assert "auc_roc" in metadata.metrics
    assert metadata.threshold == pytest.approx(model.threshold_)


def test_save_refuses_to_overwrite(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()
    name = "modelo-fixo.rapid"
    model.save(directory=tmp_path, name=name)

    with pytest.raises(FileExistsError):
        model.save(directory=tmp_path, name=name)


def test_default_filename_matches_contract():
    name = default_filename("RAPID_XGBoost", datetime(2026, 9, 2, 13, 45, 7))

    assert name == "RAPID_XGBoost-20260902-134507.rapid"


# ----------------------------------------------------------------------
# Decide (contrato §7.10)
# ----------------------------------------------------------------------

def test_decide_lists_and_selects_saved_models(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()
    model.save(directory=tmp_path, name="modelo-a.rapid")
    model.save(directory=tmp_path, name="modelo-b.rapid")

    decide = RAPID_Decide(tmp_path)

    assert {m.name for m in decide.list()} == {"modelo-a.rapid", "modelo-b.rapid"}
    assert [m.name for m in decide.execute(include=["modelo-a.rapid"])] == ["modelo-a.rapid"]
    assert [m.name for m in decide.execute(exclude=["modelo-a.rapid"])] == ["modelo-b.rapid"]
    assert len(decide.execute()) == 2


def test_decide_rejects_include_and_exclude_together(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()
    model.decide()
    model.save(directory=tmp_path, name="modelo-a.rapid")

    with pytest.raises(ValueError):
        RAPID_Decide(tmp_path).execute(include=["modelo-a.rapid"], exclude=["modelo-a.rapid"])


def test_decide_rejects_unknown_model_name(tmp_path):
    with pytest.raises(ValueError):
        RAPID_Decide(tmp_path).execute(include=["nao-existe.rapid"])


# ----------------------------------------------------------------------
# validation() — as cinco técnicas do contrato §7.9
# ----------------------------------------------------------------------

def test_validation_defaults_to_bootstrap():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    results = model.validation(n_iterations=5)

    assert set(results) == {"bootstrap"}


def test_validation_runs_requested_techniques():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    results = model.validation(
        bootstrap=True, n_iterations=5, sensitivity=True, net_benefit=True,
        external_data=_make_df(n=60, seed=9)[["x1", "x2", "desfecho"]],
    )

    assert set(results) == {"bootstrap", "external", "sensitivity", "net_benefit"}
    assert isinstance(results["net_benefit"], pd.DataFrame)


def test_fit_can_chain_validation():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit(validation={"bootstrap": True, "n_iterations": 5})

    assert model.is_validated


# ----------------------------------------------------------------------
# Decision Curve Analysis
# ----------------------------------------------------------------------

def test_dca_returns_aggregated_curve():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0.2, 0.15, 200), 0.01, 0.99)

    curve = decision_curve_analysis(y_true, y_proba, thresholds=[0.1, 0.3, 0.5])

    assert list(curve["threshold"]) == [0.1, 0.3, 0.5]
    assert (curve["net_benefit_treat_none"] == 0).all()
    # Um modelo informativo supera "tratar todos" em algum ponto da curva.
    assert (curve["net_benefit_model"] > curve["net_benefit_treat_all"]).any()


def test_dca_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        decision_curve_analysis([0, 1, 0], [0.5, 0.5])


# ----------------------------------------------------------------------
# report() (FR014)
# ----------------------------------------------------------------------

def _reportable_forest(**overrides):
    """Random Forest com amostra suficiente para a agregação SHAP sobreviver."""
    return RAPID_RandomForest(**{
        **_base_kwargs(_make_df(n=400, seed=5)),
        "n_estimators_grid": [10], "max_depth_grid": [3],
        "shap_min_bin_size": 5, **overrides,
    })


def test_report_creates_versioned_json_and_plots(tmp_path):
    model = _reportable_forest()
    model.fit()
    model.validation(bootstrap=True, n_iterations=5)
    model.decide(justification="modelo escolhido para o relatório")

    produced = model.report(output_dir=tmp_path)
    payload = json.loads(Path(produced["json"]).read_text(encoding="utf-8"))

    assert Path(produced["directory"]).exists()
    assert payload["model_type"] == "RAPID_RandomForest"
    assert payload["state"]["is_decided"] is True
    assert payload["decision"]["justification"] == "modelo escolhido para o relatório"
    assert "validation" in payload and "bootstrap" in payload["validation"]
    assert len(produced["plots"]) == 2
    assert all(Path(p).exists() for p in produced["plots"])


def test_report_json_includes_aggregated_shap_without_plots(tmp_path):
    """O SHAP agregado precisa entrar no JSON mesmo quando nenhum PNG é gerado."""
    model = _reportable_forest()
    model.fit()

    produced = model.report(format="json", output_dir=tmp_path)
    payload = json.loads(Path(produced["json"]).read_text(encoding="utf-8"))

    assert produced["plots"] == []
    assert "shap_aggregate" in payload


def test_report_skips_shap_plot_when_suppression_leaves_no_bins(tmp_path):
    """Proteger a privacidade não pode derrubar o relatório inteiro."""
    model = RAPID_RandomForest(**{
        **_base_kwargs(), "n_estimators_grid": [10], "max_depth_grid": [3],
        "shap_min_bin_size": 10_000,
    })
    model.fit()

    produced = model.report(output_dir=tmp_path)

    assert produced["json"] is not None
    assert any("shap_beeswarm" in note for note in produced["skipped"])


def test_report_is_immutable_across_calls(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    first = model.report(format="json", output_dir=tmp_path)["directory"]
    second = model.report(format="json", output_dir=tmp_path)["directory"]

    assert first != second, "cada relatório deve ir para um diretório próprio"
    assert Path(first).exists() and Path(second).exists()


def test_report_rejects_unknown_format(tmp_path):
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    with pytest.raises(ValueError):
        model.report(format="pdf", output_dir=tmp_path)


# ----------------------------------------------------------------------
# Assinaturas alinhadas ao contrato (§7.6, §7.7)
# ----------------------------------------------------------------------

def test_fit_accepts_contract_parameters():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit(cross_validation=True, k_folds=2, repetitions=1, metrics=["auc_roc"])

    assert model.cv_splits == 2
    assert model.cv_repeats == 1
    assert model.reported_metrics == ["auc_roc"]


def test_summary_accepts_table_format():
    model = _tiny_model(RAPID_LogisticL2)
    model.fit()

    model.summary(table_format="short")
    model.summary(table_format="full")
