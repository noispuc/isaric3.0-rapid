"""
Heatmap visualization for the RAPID methodology.

This module provides functions to generate heatmap plots (Step 6.4 of
the RAPID methodology). Heatmaps encode values in a matrix by color
intensity, useful for visualizing correlation and confusion matrices.

Techniques:
- correlation_heatmap: Visualize correlation between predictors.
- confusion_matrix_heatmap: Visualize classification confusion matrix.
- lca_profile_heatmap: Visualize LCA class profiles.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import confusion_matrix


def correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Heatmap",
    colorscale: str = "RdBu_r",
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Generate a correlation heatmap for numeric predictors.

    Visualizes the magnitude and direction of the correlation
    coefficient between all pairs of predictors.

    Args:
        data: Input DataFrame with numeric columns.
        title: Plot title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for correlation heatmap.")

    corr_matrix = data[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=colorscale,
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    colorscale: str = "Blues",
    height: int = 500,
    width: int = 500
) -> go.Figure:
    """
    Generate a confusion matrix heatmap for classification models.

    Displays counts or percentages of True Positives, False Positives,
    etc., as a heatmap.

    Args:
        y_true: True binary labels.
        y_pred: Predicted class labels.
        class_names: List of class names (default: ['Negative', 'Positive']).
        title: Plot title.
        normalize: If True, normalize by row (true label).
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    if class_names is None:
        class_names = ['Negative', 'Positive']

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text = np.round(cm, 2).astype(str)
        colorbar_title = "Proportion"
    else:
        text = cm.astype(str)
        colorbar_title = "Count"

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale=colorscale,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=16),
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>',
        colorbar=dict(title=colorbar_title)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='True',
        height=height,
        width=width,
        template='plotly_white',
        yaxis=dict(autorange='reversed')
    )

    return fig


def lca_profile_heatmap(
    class_profiles: pd.DataFrame,
    n_components: Optional[int] = None,
    title: str = "LCA Class Profiles",
    colorscale: str = "Viridis",
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Generate a heatmap showing LCA class profiles.

    Visualizes the probability of each feature per latent class.

    Args:
        class_profiles: DataFrame with shape (n_classes, n_features)
            containing probabilities in [0, 1].
        n_components: Number of latent classes (optional, for title).
        title: Plot title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.
    """
    y_labels = [f'Class {i}' for i in range(len(class_profiles))]

    fig = go.Figure(data=go.Heatmap(
        z=class_profiles.values,
        x=class_profiles.columns.tolist(),
        y=y_labels,
        colorscale=colorscale,
        text=np.round(class_profiles.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10),
        hovertemplate='Feature: %{x}<br>Class: %{y}<br>Probability: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Probability")
    ))

    if n_components:
        title = f"LCA Class Profiles (K={n_components})"

    fig.update_layout(
        title=title,
        xaxis_title="Variables",
        yaxis_title="Latent Class",
        height=height,
        width=width,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig