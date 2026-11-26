"""
Pytest fixtures for LCA baseline and regression testing.
"""

from pathlib import Path
from typing import List
import pandas as pd
import pytest

import isaric as isa
from tests.lca.synthetic_data import LCA_SYMPTOM_VARS, generate_synthetic_lca_data


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the root directory of the local RAPID repository."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def symptom_vars() -> List[str]:
    """Return the standard 13 binary symptom variables used in LCA tests."""
    return list(LCA_SYMPTOM_VARS)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Return a deterministic synthetic binary symptom DataFrame (N=150, seed=42)."""
    return generate_synthetic_lca_data(n_samples=150, seed=42, include_outcome=True)


@pytest.fixture(scope="module")
def model_no_bootstrap(synthetic_df: pd.DataFrame, symptom_vars: List[str]):
    """
    Scenario A fixture: LCA model trained with k=[1, 2, 3, 4] and bootstrap=False.
    """
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(
        bootstrap=False,
        param_grid={"n_clusters": [1, 2, 3, 4]},
    )
    return model


@pytest.fixture(scope="module")
def model_bootstrap(synthetic_df: pd.DataFrame, symptom_vars: List[str]):
    """
    Scenario B fixture: LCA model trained with k=[1, 2, 3, 4], bootstrap=True,
    and n_iterations=6 for fast technical execution.
    """
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(
        bootstrap=True,
        n_iterations=6,
        param_grid={"n_clusters": [1, 2, 3, 4]},
    )
    return model
