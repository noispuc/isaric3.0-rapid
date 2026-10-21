"""
Tests for Issue 7a: vectorized F1 threshold selection.

Verifies:
- Vectorized result matches the naive loop on a small dataset.
- Runtime on 126k rows is acceptable (< 1 s).
"""

import time
import numpy as np
import pytest
from sklearn.metrics import f1_score

from isaric.modelevaluation.metrics import select_classification_threshold


# ---------------------------------------------------------------------------
# Reference: the old O(n²) implementation
# ---------------------------------------------------------------------------

def _select_threshold_loop(y_true, y_prob):
    from sklearn.metrics import roc_curve, f1_score as _f1
    _, _, thresholds = roc_curve(y_true, y_prob)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = _f1(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_binary():
    rng = np.random.default_rng(0)
    n = 200
    y_prob = rng.uniform(0, 1, n)
    y_true = (y_prob + rng.normal(0, 0.3, n) > 0.5).astype(int)
    return y_true, y_prob


@pytest.fixture
def imbalanced_binary():
    """3% positive rate, similar to the dengue dataset."""
    rng = np.random.default_rng(7)
    n = 500
    y_true = (rng.uniform(0, 1, n) < 0.03).astype(int)
    y_prob = np.where(y_true == 1,
                      rng.uniform(0.1, 0.9, n),
                      rng.uniform(0.0, 0.3, n))
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Correctness: vectorized == loop
# ---------------------------------------------------------------------------

def test_matches_loop_balanced(small_binary):
    y_true, y_prob = small_binary
    t_vec = select_classification_threshold(y_true, y_prob, method="f1")
    t_loop = _select_threshold_loop(y_true, y_prob)
    # Both should yield the same F1 score at their chosen thresholds
    f1_vec = f1_score(y_true, (y_prob >= t_vec).astype(int), zero_division=0)
    f1_loop = f1_score(y_true, (y_prob >= t_loop).astype(int), zero_division=0)
    assert abs(f1_vec - f1_loop) < 1e-9, (
        f"F1 mismatch: vectorized={f1_vec:.6f} vs loop={f1_loop:.6f}"
    )


def test_matches_loop_imbalanced(imbalanced_binary):
    y_true, y_prob = imbalanced_binary
    t_vec = select_classification_threshold(y_true, y_prob, method="f1")
    t_loop = _select_threshold_loop(y_true, y_prob)
    f1_vec = f1_score(y_true, (y_prob >= t_vec).astype(int), zero_division=0)
    f1_loop = f1_score(y_true, (y_prob >= t_loop).astype(int), zero_division=0)
    assert abs(f1_vec - f1_loop) < 1e-9


def test_youden_unchanged(small_binary):
    """Youden method should still work (no regression)."""
    y_true, y_prob = small_binary
    t = select_classification_threshold(y_true, y_prob, method="youden")
    assert 0.0 <= t <= 1.0


def test_invalid_method_raises(small_binary):
    y_true, y_prob = small_binary
    with pytest.raises(ValueError, match="method must be"):
        select_classification_threshold(y_true, y_prob, method="unknown")


# ---------------------------------------------------------------------------
# Performance: 126k rows, < 1 second
# ---------------------------------------------------------------------------

def test_runtime_126k_rows():
    rng = np.random.default_rng(42)
    n = 126_000
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < 0.03).astype(int)

    start = time.perf_counter()
    select_classification_threshold(y_true, y_prob, method="f1")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 126k rows"
