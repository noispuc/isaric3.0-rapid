"""
Heatmap visualization for the RAPID methodology.

This module provides functions to generate heatmap plots (Step 6.4 of
the RAPID methodology). Heatmaps encode values in a matrix by color
intensity, useful for visualizing correlation and confusion matrices.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- correlation_heatmap: Visualize correlation between predictors.
- confusion_matrix_heatmap: Visualize classification confusion matrix.
- lca_profile_heatmap: Visualize LCA class profiles.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Heatmap",
    colorscale: str = "RdBu_r",
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a correlation heatmap for numeric predictors.

    Visualizes the magnitude and direction of the correlation
    coefficient between all pairs of predictors.

    Args:
        data: Input DataFrame with numeric columns.
        title: Plot title.
        colorscale: Plotly colorscale name or Matplotlib colormap name.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If no numeric columns found or backend invalid.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for correlation heatmap.")

    corr_matrix = data[numeric_cols].corr()

    if backend == "plotly":
        return _correlation_heatmap_plotly(
            corr_matrix=corr_matrix,
            title=title,
            colorscale=colorscale,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _correlation_heatmap_matplotlib(
            corr_matrix=corr_matrix,
            title=title,
            cmap=colorscale,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    colorscale: str = "Blues",
    height: int = 500,
    width: int = 500,
    backend: str = "plotly"
):
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
        colorscale: Plotly colorscale name or Matplotlib colormap name.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If backend invalid.
    """
    if class_names is None:
        class_names = ['Negative', 'Positive']

    cm = confusion_matrix(y_true, y_pred)

    if backend == "plotly":
        return _confusion_matrix_plotly(
            cm=cm,
            class_names=class_names,
            title=title,
            normalize=normalize,
            colorscale=colorscale,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _confusion_matrix_matplotlib(
            cm=cm,
            class_names=class_names,
            title=title,
            normalize=normalize,
            cmap=colorscale,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def lca_profile_heatmap(
    class_profiles: pd.DataFrame,
    n_components: Optional[int] = None,
    title: str = "LCA Class Profiles",
    colorscale: str = "viridis",
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a heatmap showing LCA class profiles.

    Visualizes the probability of each feature per latent class.

    Args:
        class_profiles: DataFrame with shape (n_classes, n_features)
            containing probabilities in [0, 1].
        n_components: Number of latent classes (optional, for title).
        title: Plot title.
        colorscale: Plotly colorscale name or Matplotlib colormap name.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If backend invalid.
    """
    if backend == "plotly":
        return _lca_profile_plotly(
            class_profiles=class_profiles,
            n_components=n_components,
            title=title,
            colorscale=colorscale,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _lca_profile_matplotlib(
            class_profiles=class_profiles,
            n_components=n_components,
            title=title,
            cmap=colorscale,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def lca_radar_plot(
    class_profiles: pd.DataFrame,
    n_components: Optional[int] = None,
    title: str = "LCA Class Profiles (Radar)",
    colors: Optional[List[str]] = None,
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a radar plot showing LCA class profiles.

    Each latent class is a polygon, and each axis is a feature.
    Useful for visually comparing class profiles.

    Args:
        class_profiles: DataFrame with shape (n_classes, n_features)
            containing probabilities in [0, 1].
        n_components: Number of latent classes (optional, for title).
        title: Plot title.
        colors: List of colors for each class.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if backend == "plotly":
        return _lca_radar_plotly(
            class_profiles, n_components, title, colors, height, width
        )
    elif backend == "matplotlib":
        return _lca_radar_matplotlib(
            class_profiles, n_components, title, colors, height, width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")

# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _correlation_heatmap_plotly(
    corr_matrix: pd.DataFrame,
    title: str,
    colorscale: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly correlation heatmap."""
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


def _confusion_matrix_plotly(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    normalize: bool,
    colorscale: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        text = np.round(cm_display, 2).astype(str)
        colorbar_title = "Proportion"
    else:
        cm_display = cm
        text = cm.astype(str)
        colorbar_title = "Count"

    fig = go.Figure(data=go.Heatmap(
        z=cm_display,
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


def _lca_profile_plotly(
    class_profiles: pd.DataFrame,
    n_components: Optional[int],
    title: str,
    colorscale: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly LCA profile heatmap."""
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


def _lca_radar_plotly(
    class_profiles: pd.DataFrame,
    n_components: Optional[int],
    title: str,
    colors: Optional[List[str]],
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly radar plot for LCA profiles."""
    features = class_profiles.columns.tolist()
    n_features = len(features)

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261',
                  '#8a5a44', '#a8dadc', '#457b9d', '#e63946', '#06d6a0']

    fig = go.Figure()

    for i, (class_name, row) in enumerate(class_profiles.iterrows()):
        # Fecha o polígono (repetir o primeiro valor no final)
        values = row.values.tolist() + [row.values[0]]
        angles = features + [features[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=angles,
            fill='toself',
            name=str(class_name),
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.6
        ))

    if n_components:
        title = f"LCA Class Profiles (K={n_components})"

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=title,
        height=height,
        width=width,
        template='plotly_white',
        showlegend=True
    )

    return fig

# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _correlation_heatmap_matplotlib(
    corr_matrix: pd.DataFrame,
    title: str,
    cmap: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib correlation heatmap."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns.tolist(), rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.index.tolist())
    
    # Add text annotations
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(
                j, i, f'{corr_matrix.values[i, j]:.2f}',
                ha='center', va='center',
                fontsize=8
            )
    
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Correlation')
    fig.tight_layout()
    return fig


def _confusion_matrix_matplotlib(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    normalize: bool,
    cmap: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        colorbar_title = "Proportion"
    else:
        cm_display = cm
        colorbar_title = "Count"
    
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    im = ax.imshow(cm_display, cmap=cmap)
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f'{cm_display[i, j]:.2f}'
            else:
                text = f'{int(cm_display[i, j])}'
            ax.text(j, i, text, ha='center', va='center', fontsize=16)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_title)
    fig.tight_layout()
    return fig


def _lca_profile_matplotlib(
    class_profiles: pd.DataFrame,
    n_components: Optional[int],
    title: str,
    cmap: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib LCA profile heatmap."""
    y_labels = [f'Class {i}' for i in range(len(class_profiles))]
    
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    im = ax.imshow(class_profiles.values, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(class_profiles.columns)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(class_profiles.columns.tolist(), rotation=45, ha='right')
    ax.set_yticklabels(y_labels)
    
    # Add text annotations
    for i in range(len(y_labels)):
        for j in range(len(class_profiles.columns)):
            ax.text(
                j, i, f'{class_profiles.values[i, j]:.2f}',
                ha='center', va='center',
                fontsize=8
            )
    
    if n_components:
        title = f"LCA Class Profiles (K={n_components})"
    
    ax.set_xlabel('Variables')
    ax.set_ylabel('Latent Class')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Probability')
    fig.tight_layout()
    return fig

def _lca_radar_matplotlib(
    class_profiles: pd.DataFrame,
    n_components: Optional[int],
    title: str,
    colors: Optional[List[str]],
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib radar plot for LCA profiles."""
    features = class_profiles.columns.tolist()
    n_features = len(features)

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261',
                  '#8a5a44', '#a8dadc', '#457b9d', '#e63946', '#06d6a0']

    # Ângulos
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # fechar

    fig, ax = plt.subplots(figsize=(width/100, height/100),
                            subplot_kw=dict(polar=True))

    for i, (class_name, row) in enumerate(class_profiles.iterrows()):
        values = row.values.tolist() + [row.values[0]]

        ax.plot(angles, values, 'o-', linewidth=2,
                color=colors[i % len(colors)], label=str(class_name))
        ax.fill(angles, values, alpha=0.2, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    fig.tight_layout()
    return fig