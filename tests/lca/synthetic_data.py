"""
Synthetic binary symptom data generator for LCA testing in RAPID v0.2.0.

Generates reproducible binary clinical symptom profiles compatible with
the SINAN Dengue reference variable set for technical verification,
structural validation, and regression testing.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

LCA_SYMPTOM_VARS: List[str] = [
    "Fever",
    "Myalgia",
    "Headache",
    "Rash",
    "Vomiting",
    "Nausea",
    "Back pain",
    "Conjunctivitis",
    "Arthritis",
    "Arthralgia",
    "Petechiae",
    "Tourniquet test",
    "Retro-orbital pain",
]


def generate_synthetic_lca_data(
    n_samples: int = 150,
    seed: int = 42,
    symptom_vars: Optional[List[str]] = None,
    include_outcome: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic binary symptom dataset with 3 latent class profiles.

    Latent classes simulated:
    - Class 0 (Mild febrile, ~50%): High fever/headache/myalgia, low warning signs.
    - Class 1 (GI / Rash predominant, ~30%): Moderate fever, high rash/vomiting/nausea.
    - Class 2 (Hemorrhagic / Severe profile, ~20%): High arthralgia, petechiae,
      tourniquet test, retro-orbital pain, and back pain.

    Args:
        n_samples: Number of synthetic patient records (default 150).
        seed: Fixed random seed for reproducibility (default 42).
        symptom_vars: List of binary symptom names. Defaults to LCA_SYMPTOM_VARS.
        include_outcome: Whether to include a binary 'severe' outcome column
            and 'true_latent_class' metadata column.

    Returns:
        DataFrame with binary (0/1) columns for each symptom variable.
    """
    if symptom_vars is None:
        symptom_vars = list(LCA_SYMPTOM_VARS)

    rng = np.random.default_rng(seed)
    n_vars = len(symptom_vars)

    # Base probabilities for 13 standard symptoms across 3 distinct profiles
    base_profiles = np.array(
        [
            # Fever, Myal, Head, Rash, Vomit, Naus, Back, Conj, Arth, Arthr, Petec, Tourn, Retro
            [0.92, 0.78, 0.82, 0.12, 0.10, 0.15, 0.18, 0.05, 0.08, 0.20, 0.06, 0.08, 0.22],
            [0.85, 0.55, 0.65, 0.68, 0.75, 0.80, 0.30, 0.15, 0.12, 0.35, 0.18, 0.15, 0.35],
            [0.95, 0.88, 0.90, 0.55, 0.65, 0.70, 0.78, 0.35, 0.60, 0.85, 0.82, 0.78, 0.85],
        ],
        dtype=float,
    )

    # Adapt if custom symptom_vars length differs from 13
    if n_vars != base_profiles.shape[1]:
        reps = int(np.ceil(n_vars / base_profiles.shape[1]))
        base_profiles = np.tile(base_profiles, (1, reps))[:, :n_vars]

    class_proportions = [0.50, 0.30, 0.20]
    latent_classes = rng.choice([0, 1, 2], size=n_samples, p=class_proportions)

    probs = base_profiles[latent_classes]
    binary_matrix = rng.binomial(1, probs).astype(int)

    # Guarantee no constant (zero-variance) columns even for small n_samples
    for col_idx in range(n_vars):
        col = binary_matrix[:, col_idx]
        if col.sum() == 0:
            binary_matrix[0, col_idx] = 1
        elif col.sum() == n_samples:
            binary_matrix[0, col_idx] = 0

    df = pd.DataFrame(binary_matrix, columns=symptom_vars)

    if include_outcome:
        outcome_probs = np.where(
            latent_classes == 2, 0.75, np.where(latent_classes == 1, 0.20, 0.05)
        )
        df["severe"] = rng.binomial(1, outcome_probs).astype(int)

    return df
