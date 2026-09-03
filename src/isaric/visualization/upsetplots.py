"""
Upset plot visualization for the RAPID methodology.

This module provides functions to generate upset plots (Step 6.2 of
the RAPID methodology). Upset plots represent intersections of
multiple sets, useful for visualizing co-occurring comorbidities
or complications.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- upset_plot: Visualize intersections of multiple binary features.
- set_size_plot: Show total size of each individual set.
- intersection_size_plot: Show frequency of unique combinations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def upset_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Upset Plot",
    height: int = 600,
    width: int = 900,
    backend: str = "plotly"
):
    """
    Generate an upset plot for multiple binary features.

    Shows the frequency of each unique intersection combination and
    the total size of each set.

    Args:
        data: Input DataFrame with binary columns (0/1 or True/False).
        set_columns: List of binary columns representing sets.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If columns are not found, not binary, or backend invalid.
    """
    _validate_binary_columns(data, set_columns)

    # Calculate intersections
    data_copy = data[set_columns].astype(bool)
    data_copy['_combination'] = data_copy.apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    intersection_counts = data_copy['_combination'].value_counts().sort_index()
    set_sizes = data_copy[set_columns].sum()

    # Create labels for combinations
    labels = []
    for combo in intersection_counts.index:
        parts = []
        for i, val in enumerate(combo):
            if val == '1':
                parts.append(set_columns[i])
        labels.append(' & '.join(parts) if parts else 'None')

    if backend == "plotly":
        return _upset_plotly(
            intersection_counts=intersection_counts,
            set_sizes=set_sizes,
            labels=labels,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _upset_matplotlib(
            intersection_counts=intersection_counts,
            set_sizes=set_sizes,
            labels=labels,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def set_size_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Set Sizes",
    height: int = 400,
    width: int = 500,
    backend: str = "plotly"
):
    """
    Generate a horizontal bar plot showing the size of each set.

    Args:
        data: Input DataFrame with binary columns.
        set_columns: List of binary columns representing sets.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If columns are not found, not binary, or backend invalid.
    """
    _validate_binary_columns(data, set_columns)

    set_sizes = data[set_columns].sum().sort_values(ascending=True)

    if backend == "plotly":
        return _set_size_plotly(
            set_sizes=set_sizes,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _set_size_matplotlib(
            set_sizes=set_sizes,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def intersection_size_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Intersection Sizes",
    height: int = 400,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a bar plot showing frequency of each intersection combination.

    Args:
        data: Input DataFrame with binary columns.
        set_columns: List of binary columns representing sets.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If columns are not found, not binary, or backend invalid.
    """
    _validate_binary_columns(data, set_columns)

    data_copy = data[set_columns].astype(bool)
    data_copy['_combination'] = data_copy.apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    intersection_counts = data_copy['_combination'].value_counts().sort_index()

    labels = []
    for combo in intersection_counts.index:
        parts = []
        for i, val in enumerate(combo):
            if val == '1':
                parts.append(set_columns[i])
        labels.append(' & '.join(parts) if parts else 'None')

    if backend == "plotly":
        return _intersection_size_plotly(
            intersection_counts=intersection_counts,
            labels=labels,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _intersection_size_matplotlib(
            intersection_counts=intersection_counts,
            labels=labels,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _upset_plotly(
    intersection_counts: pd.Series,
    set_sizes: pd.Series,
    labels: List[str],
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly upset plot."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )

    fig.add_trace(
        go.Bar(
            x=list(range(len(intersection_counts))),
            y=intersection_counts.values,
            name='Intersection Size',
            marker_color='#2a9d8f',
            text=intersection_counts.values,
            textposition='outside',
            hovertemplate='Intersection: %{x}<br>Count: %{y}<extra></extra>'
        ),
        row=1,
        col=1
    )

    sorted_sizes = set_sizes.sort_values(ascending=True)
    fig.add_trace(
        go.Bar(
            x=sorted_sizes.values,
            y=sorted_sizes.index.tolist(),
            orientation='h',
            name='Set Size',
            marker_color='#264653',
            text=sorted_sizes.values,
            textposition='outside',
            hovertemplate='Set: %{y}<br>Size: %{x}<extra></extra>'
        ),
        row=2,
        col=1
    )

    fig.update_xaxes(
        tickvals=list(range(len(intersection_counts))),
        ticktext=labels,
        tickangle=-45,
        row=1,
        col=1
    )

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white',
        showlegend=False
    )

    return fig


def _set_size_plotly(
    set_sizes: pd.Series,
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly set size plot."""
    fig = go.Figure(data=go.Bar(
        x=set_sizes.values,
        y=set_sizes.index.tolist(),
        orientation='h',
        marker_color='#2a9d8f',
        text=set_sizes.values,
        textposition='outside',
        hovertemplate='Set: %{y}<br>Size: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Number of Observations',
        yaxis_title='Set',
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _intersection_size_plotly(
    intersection_counts: pd.Series,
    labels: List[str],
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly intersection size plot."""
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=intersection_counts.values,
        marker_color='#2a9d8f',
        text=intersection_counts.values,
        textposition='outside',
        hovertemplate='Intersection: %{x}<br>Count: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Intersection',
        yaxis_title='Count',
        height=height,
        width=width,
        template='plotly_white',
        xaxis_tickangle=-45
    )

    return fig


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _upset_matplotlib(
    intersection_counts: pd.Series,
    set_sizes: pd.Series,
    labels: List[str],
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib upset plot."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(width/100, height/100),
        gridspec_kw={'height_ratios': [0.7, 0.3]},
        sharex=False
    )

    # Intersection size bar chart
    bars = ax1.bar(
        range(len(intersection_counts)),
        intersection_counts.values,
        color='#2a9d8f'
    )
    ax1.set_xticks(range(len(intersection_counts)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Intersection Size')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=8
        )

    # Set size horizontal bar chart
    sorted_sizes = set_sizes.sort_values(ascending=True)
    bars2 = ax2.barh(
        sorted_sizes.index.tolist(),
        sorted_sizes.values,
        color='#264653'
    )
    ax2.set_xlabel('Set Size')
    
    # Add value labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(
            width,
            bar.get_y() + bar.get_height()/2.,
            f'{int(width)}',
            ha='left',
            va='center',
            fontsize=8
        )

    ax1.set_title(title)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _set_size_matplotlib(
    set_sizes: pd.Series,
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib set size plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    bars = ax.barh(
        set_sizes.index.tolist(),
        set_sizes.values,
        color='#2a9d8f'
    )
    
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height()/2.,
            f'{int(width)}',
            ha='left',
            va='center',
            fontsize=10
        )
    
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Set')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _intersection_size_matplotlib(
    intersection_counts: pd.Series,
    labels: List[str],
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib intersection size plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    bars = ax.bar(
        range(len(intersection_counts)),
        intersection_counts.values,
        color='#2a9d8f'
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    ax.set_xticks(range(len(intersection_counts)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Intersection')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _validate_binary_columns(
    data: pd.DataFrame,
    columns: List[str]
) -> None:
    """
    Validate that columns are binary (0/1 or True/False).

    Args:
        data: Input DataFrame.
        columns: List of column names.

    Raises:
        ValueError: If columns are not found or are not binary.
    """
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        unique_values = data[col].dropna().unique()
        valid_binary = {0, 1, 0.0, 1.0, True, False}

        if not set(unique_values).issubset(valid_binary):
            raise ValueError(
                f"Column '{col}' must be binary (0/1 or True/False). "
                f"Found values: {unique_values}"
            )