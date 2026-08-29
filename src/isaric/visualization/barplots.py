"""
Bar plot visualization for the RAPID methodology.

This module provides functions to generate bar plots (Step 6.1 of
the RAPID methodology). Bar plots compare quantities across discrete
categories.

Techniques:
- simple_bar_plot: Basic bar plot for counts or frequencies.
- grouped_bar_plot: Compare multiple groups side by side.
- stacked_bar_plot: Display relative proportions across groups.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import plotly.graph_objs as go
import plotly.express as px


def simple_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    title: str = "Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a simple bar plot.

    If y_col is None, counts the frequency of each category in x_col.

    Args:
        data: Input DataFrame.
        x_col: Column for categories (x-axis).
        y_col: Column for values (y-axis). If None, counts frequencies.
        title: Plot title.
        xaxis_title: X-axis label (defaults to x_col).
        yaxis_title: Y-axis label (defaults to y_col or "Count").
        color: Bar color.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If x_col is not found in DataFrame.
    """
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    if y_col is None:
        # Count frequencies
        value_counts = data[x_col].value_counts().reset_index()
        value_counts.columns = [x_col, 'Count']
        x = value_counts[x_col]
        y = value_counts['Count']
        y_label = yaxis_title or 'Count'
    else:
        if y_col not in data.columns:
            raise ValueError(f"Column '{y_col}' not found in DataFrame.")
        x = data[x_col]
        y = data[y_col]
        y_label = yaxis_title or y_col

    fig = go.Figure(data=go.Bar(
        x=x,
        y=y,
        marker_color=color,
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=y_label,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def grouped_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "Grouped Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a grouped bar plot comparing multiple groups.

    Displays bars for each group side by side for comparison.

    Args:
        data: Input DataFrame.
        x_col: Column for categories (x-axis).
        y_col: Column for values (y-axis).
        group_col: Column defining the groups to compare.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    for col in [x_col, y_col, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        color=group_col,
        barmode='group',
        title=title,
        height=height,
        width=width,
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or y_col
    )

    return fig


def stacked_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "Stacked Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a stacked bar plot showing relative proportions.

    Stacks bars for each group to show the composition of categories.

    Args:
        data: Input DataFrame.
        x_col: Column for categories (x-axis).
        y_col: Column for values (y-axis).
        group_col: Column defining the groups to stack.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    for col in [x_col, y_col, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        color=group_col,
        barmode='stack',
        title=title,
        height=height,
        width=width,
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or y_col
    )

    return fig