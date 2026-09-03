"""
Bar plot visualization for the RAPID methodology.

This module provides functions to generate bar plots (Step 6.1 of
the RAPID methodology). Bar plots compare quantities across discrete
categories.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

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
import matplotlib.pyplot as plt


def simple_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: Optional[str] = None,
    title: str = "Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If x_col is not found or backend invalid.
    """
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    if y_col is None:
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

    if backend == "plotly":
        return _simple_bar_plotly(
            x=x,
            y=y,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=y_label,
            color=color,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _simple_bar_matplotlib(
            x=x,
            y=y,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=y_label,
            color=color,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def grouped_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "Grouped Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [x_col, y_col, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _grouped_bar_plotly(
            data=data,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _grouped_bar_matplotlib(
            data=data,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def stacked_bar_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = "Stacked Bar Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [x_col, y_col, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _stacked_bar_plotly(
            data=data,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _stacked_bar_matplotlib(
            data=data,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _simple_bar_plotly(
    x,
    y,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly simple bar plot."""
    fig = go.Figure(data=go.Bar(
        x=x,
        y=y,
        marker_color=color,
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _grouped_bar_plotly(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly grouped bar plot."""
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
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )

    return fig


def _stacked_bar_plotly(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly stacked bar plot."""
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
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title
    )

    return fig


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _simple_bar_matplotlib(
    x,
    y,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib simple bar plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    bars = ax.bar(x, y, color=color)
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    
    # Rotate x labels if many categories
    if len(x) > 5:
        ax.tick_params(axis='x', rotation=45)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _grouped_bar_matplotlib(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib grouped bar plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    categories = data[x_col].unique()
    groups = data[group_col].unique()
    n_groups = len(groups)
    
    x_positions = np.arange(len(categories))
    bar_width = 0.8 / n_groups
    
    for i, group in enumerate(groups):
        group_data = data[data[group_col] == group]
        values = [group_data[group_data[x_col] == cat][y_col].values[0] 
                  if len(group_data[group_data[x_col] == cat]) > 0 else 0 
                  for cat in categories]
        
        ax.bar(
            x_positions + i * bar_width,
            values,
            bar_width,
            label=str(group)
        )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.set_xticks(x_positions + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _stacked_bar_matplotlib(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib stacked bar plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    categories = data[x_col].unique()
    groups = data[group_col].unique()
    
    bottom = np.zeros(len(categories))
    
    for group in groups:
        group_data = data[data[group_col] == group]
        values = [group_data[group_data[x_col] == cat][y_col].values[0] 
                  if len(group_data[group_data[x_col] == cat]) > 0 else 0 
                  for cat in categories]
        
        ax.bar(
            categories,
            values,
            bottom=bottom,
            label=str(group)
        )
        bottom += values
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig