"""
Tests for Issues 4 + 6:
  - Issue 4: metrics computed on hold-out test set, configurable threshold.
  - Issue 6: preprocessing transformers fit only on train, no leakage to test.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import isaric as isa
from isaric.preprocessing.transformers import (
    RAPIDStandardScaler,
    RAPIDOneHotEncoder,
    RAPIDImputer,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "outcome": rng.integers(0, 2, n),
    })


FEATS = ["x1", "x2"]


# ---------------------------------------------------------------------------
# Issue 4a: split attributes exist after create()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["lasso", "ridge", "elastic_net",
                                        "decision_tree", "random_forest"])
def test_split_attributes_set_after_create(small_df, model_key):
    inst = isa.RAPID.create(
        data=small_df, model=model_key,
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    for attr in ("train_idx", "test_idx", "X_train", "X_test", "y_train", "y_test"):
        assert hasattr(inst, attr), f"Missing attribute '{attr}' after create()"


@pytest.mark.parametrize("model_key", ["lasso", "ridge"])
def test_split_sizes_70_30(small_df, model_key):
    inst = isa.RAPID.create(
        data=small_df, model=model_key,
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    n = len(small_df)
    assert len(inst.train_idx) + len(inst.test_idx) == n
    assert abs(len(inst.test_idx) / n - 0.3) < 0.05


@pytest.mark.parametrize("model_key", ["lasso", "ridge"])
def test_split_stratified(small_df, model_key):
    inst = isa.RAPID.create(
        data=small_df, model=model_key,
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    train_prev = inst.y_train.mean()
    test_prev  = inst.y_test.mean()
    full_prev  = small_df["outcome"].mean()
    assert abs(train_prev - full_prev) < 0.05
    assert abs(test_prev  - full_prev) < 0.05


# ---------------------------------------------------------------------------
# Issue 4b: metrics after fit() are evaluated on test set, not train set
# ---------------------------------------------------------------------------

def _make_leaky_df(n_train=150, n_test=50, seed=0):
    """Dataset where train AUC and test AUC differ noticeably."""
    rng = np.random.default_rng(seed)
    # Train: strong signal (x1 and outcome highly correlated)
    x1_train = rng.standard_normal(n_train)
    y_train  = (x1_train + rng.normal(0, 0.3, n_train) > 0).astype(int)
    # Test: weaker signal
    x1_test  = rng.standard_normal(n_test)
    y_test   = (x1_test + rng.normal(0, 1.2, n_test) > 0).astype(int)
    x2_train = rng.standard_normal(n_train)
    x2_test  = rng.standard_normal(n_test)
    train_df = pd.DataFrame({"x1": x1_train, "x2": x2_train, "outcome": y_train})
    test_df  = pd.DataFrame({"x1": x1_test,  "x2": x2_test,  "outcome": y_test})
    return pd.concat([train_df, test_df], ignore_index=True), n_train


leaky_df, n_train = _make_leaky_df()


def test_metrics_on_test_set_not_train(small_df):
    """
    Metrics stored in instance.metrics after fit() must match those computed
    directly on X_test/y_test, NOT on X_train/y_train.
    """
    inst = isa.RAPID.create(
        data=small_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
        test_size=0.3, random_seed=42,
    )
    inst.fit()

    # Recompute AUC manually on test set
    y_prob_test = inst.fitted_model.predict_proba(inst.X_test)[:, 1]
    auc_test = roc_auc_score(inst.y_test, y_prob_test)

    # Recompute AUC on train set (should be different / higher)
    y_prob_train = inst.fitted_model.predict_proba(inst.X_train)[:, 1]
    auc_train = roc_auc_score(inst.y_train, y_prob_train)

    reported_auc = inst.metrics["auc"]

    # Reported AUC must match test AUC, not train AUC
    assert abs(reported_auc - auc_test) < 1e-9, (
        f"Reported AUC ({reported_auc:.4f}) does not match test AUC ({auc_test:.4f})"
    )
    # Train AUC may differ — verify they can differ (sanity check on fixture)
    # (no strict assertion here — just ensure the right set is used above)


# ---------------------------------------------------------------------------
# Issue 4c: configurable threshold
# ---------------------------------------------------------------------------

def test_threshold_float_stored(small_df):
    inst = isa.RAPID.create(
        data=small_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(threshold=0.3)
    assert abs(inst._resolved_threshold - 0.3) < 1e-9


def test_threshold_f1_selected_on_train(small_df):
    inst = isa.RAPID.create(
        data=small_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(threshold="f1")
    # Threshold must be a float in (0, 1)
    t = inst._resolved_threshold
    assert isinstance(t, float)
    assert 0.0 < t < 1.0


def test_threshold_youden_selected_on_train(small_df):
    inst = isa.RAPID.create(
        data=small_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(threshold="youden")
    t = inst._resolved_threshold
    assert isinstance(t, float)
    assert 0.0 < t < 1.0


def test_f1_threshold_not_leaked_from_test(small_df):
    """
    The F1 threshold must be selected using only y_train/y_prob_train.
    Verify by checking that _resolved_threshold matches the value computed
    on train alone.
    """
    from isaric.modelevaluation.metrics import select_classification_threshold

    inst = isa.RAPID.create(
        data=small_df, model="lasso",
        dependent_var="outcome", independent_vars=FEATS,
    )
    inst.fit(threshold="f1")

    # Independently compute threshold on train only
    y_prob_train = inst.fitted_model.predict_proba(inst.X_train)[:, 1]
    expected_t = select_classification_threshold(
        inst.y_train.values, y_prob_train, method="f1"
    )
    assert abs(inst._resolved_threshold - expected_t) < 1e-9


# ---------------------------------------------------------------------------
# Issue 6: RAPIDStandardScaler — no leakage
# ---------------------------------------------------------------------------

def test_standard_scaler_fit_transform_uses_train_stats():
    """Scaler fitted on X_train: mean_ must equal X_train mean, not global."""
    rng = np.random.default_rng(99)
    X_train = pd.DataFrame({"a": rng.normal(10, 1, 100), "b": rng.normal(-5, 2, 100)})
    X_test  = pd.DataFrame({"a": rng.normal(20, 1, 50),  "b": rng.normal(5, 2, 50)})

    scaler = RAPIDStandardScaler()
    scaler.fit(X_train)

    # mean_ from scaler must match X_train mean
    np.testing.assert_allclose(scaler.mean_, X_train.mean().values, rtol=1e-5)

    # After transforming test, values should NOT center around zero
    # (since X_test mean is shifted relative to train)
    X_test_scaled = scaler.transform(X_test)
    assert abs(X_test_scaled["a"].mean()) > 3.0, (
        "Test set 'a' should be far from zero when fitted on a different distribution"
    )


def test_standard_scaler_mean_not_contaminated_by_test():
    """Verify scaler mean_ matches only train distribution."""
    rng = np.random.default_rng(7)
    X_train = pd.DataFrame({"x": rng.normal(0, 1, 200)})
    X_test  = pd.DataFrame({"x": rng.normal(100, 1, 50)})

    scaler = RAPIDStandardScaler()
    scaler.fit(X_train)
    global_mean = pd.concat([X_train, X_test])["x"].mean()

    assert abs(scaler.mean_[0] - X_train["x"].mean()) < 0.2
    assert abs(scaler.mean_[0] - global_mean) > 10, (
        "Scaler was contaminated by test distribution"
    )


def test_standard_scaler_fit_transform_equivalent(small_df):
    X = small_df[FEATS]
    scaler = RAPIDStandardScaler()
    result_ft  = scaler.fit_transform(X)
    scaler2 = RAPIDStandardScaler()
    result_sep = scaler2.fit(X).transform(X)
    pd.testing.assert_frame_equal(result_ft, result_sep)


def test_standard_scaler_transform_without_fit_raises():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="fit"):
        RAPIDStandardScaler().transform(X)


# ---------------------------------------------------------------------------
# Issue 6: RAPIDOneHotEncoder — categories from train only
# ---------------------------------------------------------------------------

def test_onehot_encoder_categories_from_train():
    X_train = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    X_test  = pd.DataFrame({"cat": ["a", "c", "b", "c"]})  # 'c' unseen

    enc = RAPIDOneHotEncoder(drop_first=False)
    enc.fit(X_train)
    X_train_enc = enc.transform(X_train)
    X_test_enc  = enc.transform(X_test)

    # Only categories from train appear as columns
    assert set(X_train_enc.columns) == set(X_test_enc.columns)
    # 'c' was unseen in train — must not create a cat_c column
    assert "cat_c" not in X_test_enc.columns
    assert "cat_c" not in X_train_enc.columns


# ---------------------------------------------------------------------------
# Issue 6: RAPIDImputer — imputation values from train only
# ---------------------------------------------------------------------------

def test_imputer_fit_transform_shape():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(1, 1, 50)})
    X.iloc[::5, 0] = np.nan  # introduce missings

    imp = RAPIDImputer(n=2, max_iter=2)
    X_imp = imp.fit_transform(X)

    assert X_imp.shape == X.shape
    assert X_imp.isna().sum().sum() == 0


def test_imputer_transform_without_fit_raises():
    X = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(RuntimeError, match="fit"):
        RAPIDImputer().transform(X)
