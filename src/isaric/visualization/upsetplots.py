"""
Upset plot visualization for the RAPID methodology.

This module provides functions to generate upset plots (Step 6.2 of
the RAPID methodology). Upset plots represent intersections of
multiple sets, useful for visualizing co-occurring comorbidities
or complications.

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


def upset_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Upset Plot",
    height: int = 600,
    width: int = 900
) -> go.Figure:
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

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If columns are not found or are not binary.
    """
    _validate_binary_columns(data, set_columns)

    # Calculate intersections
    data_copy = data[set_columns].astype(bool)
    data_copy['_combination'] = data_copy.apply(
        lambda row: ''.join(row.astype(int).astype(str)), axis=1
    )
    intersection_counts = data_copy['_combination'].value_counts().sort_index()

    # Set sizes
    set_sizes = data_copy[set_columns].sum()

    # Create labels for combinations
    labels = []
    for combo in intersection_counts.index:
        parts = []
        for i, val in enumerate(combo):
            if val == '1':
                parts.append(set_columns[i])
        labels.append(' & '.join(parts) if parts else 'None')

    # Build figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )

    # Intersection size bar chart
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

    # Set size bar chart (horizontal)
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

    # X-axis labels for intersections
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


def set_size_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Set Sizes",
    height: int = 400,
    width: int = 500
) -> go.Figure:
    """
    Generate a horizontal bar plot showing the size of each set.

    Args:
        data: Input DataFrame with binary columns.
        set_columns: List of binary columns representing sets.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If columns are not found or are not binary.
    """
    _validate_binary_columns(data, set_columns)

    set_sizes = data[set_columns].sum().sort_values(ascending=True)

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


def intersection_size_plot(
    data: pd.DataFrame,
    set_columns: List[str],
    title: str = "Intersection Sizes",
    height: int = 400,
    width: int = 700
) -> go.Figure:
    """
    Generate a bar plot showing frequency of each intersection combination.

    Args:
        data: Input DataFrame with binary columns.
        set_columns: List of binary columns representing sets.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If columns are not found or are not binary.
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