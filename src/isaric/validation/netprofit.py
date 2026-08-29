"""
Net benefit analysis for the RAPID methodology.

This module provides functions to evaluate clinical utility using
Decision Curve Analysis (DCA) (Step 5.5 of the RAPID methodology).
DCA quantifies net benefit relative to different treatment thresholds,
incorporating clinical consequences of decisions.

Techniques:
- decision_curve_analysis: Calculate net benefit across thresholds.
- net_benefit_curve: Generate decision curve plot data.
- treat_all_none: Calculate net benefit for treat-all and treat-none.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def decision_curve_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Calculate net benefit across a range of treatment thresholds.

    Net benefit = (TP / N) - (FP / N) * (threshold / (1 - threshold))

    Args:
        y_true: True binary labels (0/1).
        y_prob: Predicted probabilities.
        thresholds: List of threshold probabilities (None = 0.1 to 0.5).

    Returns:
        DataFrame with columns: Threshold, Net_Benefit_Model,
        Net_Benefit_All, Net_Benefit_None.

    Raises:
        ValueError: If y_true is not binary.
    """
    _validate_binary(y_true)

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.5, 41)  # 0.10 to 0.50 step 0.01

    n = len(y_true)
    n_positives = int(np.sum(y_true))

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))

        # Net benefit of the model
        nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))

        # Net benefit of treat all
        nb_all = (n_positives / n) - ((n - n_positives) / n) * (threshold / (1 - threshold))

        # Net benefit of treat none
        nb_none = 0.0

        results.append({
            'Threshold': round(float(threshold), 4),
            'Net_Benefit_Model': round(float(nb_model), 6),
            'Net_Benefit_All': round(float(nb_all), 6),
            'Net_Benefit_None': nb_none
        })

    return pd.DataFrame(results)


def net_benefit_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None
) -> Dict[str, np.ndarray]:
    """
    Generate data for decision curve plot.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        thresholds: List of thresholds.

    Returns:
        Dictionary with arrays for plotting.

    Raises:
        ValueError: If y_true is not binary.
    """
    _validate_binary(y_true)

    df = decision_curve_analysis(y_true, y_prob, thresholds)

    return {
        'thresholds': df['Threshold'].values,
        'net_benefit_model': df['Net_Benefit_Model'].values,
        'net_benefit_all': df['Net_Benefit_All'].values,
        'net_benefit_none': df['Net_Benefit_None'].values
    }


def treat_all_none(
    y_true: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Calculate net benefit for treat-all and treat-none strategies.

    Args:
        y_true: True binary labels.
        threshold: Treatment threshold probability.

    Returns:
        Tuple of (net_benefit_treat_all, net_benefit_treat_none).

    Raises:
        ValueError: If threshold is invalid.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"threshold must be between 0.0 and 1.0. Received: {threshold}"
        )

    n = len(y_true)
    n_positives = int(np.sum(y_true))

    nb_all = (n_positives / n) - ((n - n_positives) / n) * (threshold / (1 - threshold))
    nb_none = 0.0

    return float(nb_all), float(nb_none)


def clinical_utility_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> pd.DataFrame:
    """
    Generate clinical utility summary across all thresholds.

    This is a convenience wrapper around decision_curve_analysis that
    includes the standard treat-all and treat-none reference lines.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        DataFrame with net benefit for model, treat-all, and treat-none.

    Raises:
        ValueError: If y_true is not binary.
    """
    return decision_curve_analysis(y_true, y_prob)


def _validate_binary(y: np.ndarray) -> None:
    """
    Validate that array is binary (0/1).

    Args:
        y: Input array.

    Raises:
        ValueError: If array is not binary.
    """
    unique_values = np.unique(y)
    if len(unique_values) != 2:
        raise ValueError(
            f"y must be binary. Found {len(unique_values)} unique values."
        )
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"y must be coded as 0/1. Found: {unique_values}")