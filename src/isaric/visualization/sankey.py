"""
Sankey diagram visualization for the RAPID methodology.

This module provides functions to generate Sankey diagrams (Step 6.6
of the RAPID methodology). Sankey diagrams visualize the flow of
quantities from one set of categories to another.

Techniques:
- patient_pathway: Visualize patient transitions between states.
- cohort_flow: Display inclusion/exclusion criteria flow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objs as go


def patient_pathway(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    value_col: Optional[str] = None,
    title: str = "Patient Pathway",
    height: int = 600,
    width: int = 900
) -> go.Figure:
    """
    Generate a Sankey diagram showing patient transitions between states.

    Visualizes the flow of patients from one state (e.g., diagnosis) to
    another (e.g., treatment), showing quantities through line widths.

    Args:
        data: Input DataFrame.
        source_col: Column for source state.
        target_col: Column for target state.
        value_col: Column for flow quantity (None = count).
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    for col in [source_col, target_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Count transitions if value_col is None
    if value_col is None:
        flows = data.groupby([source_col, target_col]).size().reset_index(name='count')
        value_col = 'count'

    # Get all unique states
    all_states = list(set(
        data[source_col].unique().tolist() +
        data[target_col].unique().tolist()
    ))

    state_to_idx = {state: i for i, state in enumerate(all_states)}

    source_indices = data[source_col].map(state_to_idx).tolist()
    target_indices = data[target_col].map(state_to_idx).tolist()
    values = data[value_col].tolist()

    fig = go.Figure(data=go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_states,
            color="#2a9d8f"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>'
        )
    ))

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def cohort_flow(
    data: pd.DataFrame,
    stages: List[str],
    title: str = "Cohort Flow",
    height: int = 600,
    width: int = 900
) -> go.Figure:
    """
    Display how an initial cohort is stratified through multiple stages.

    Shows the sequential inclusion/exclusion criteria and how patient
    numbers change at each step.

    Args:
        data: Input DataFrame.
        stages: List of column names representing sequential stages.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If stages are not found or are not binary.
    """
    for col in stages:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    labels = []
    source_indices = []
    target_indices = []
    values = []

    current_count = len(data)
    previous_label = "Initial Cohort"

    labels.append(previous_label)

    for i, stage in enumerate(stages):
        stage_count = int(data[stage].sum())
        stage_label = f"{stage} (n={stage_count})"

        labels.append(stage_label)
        source_indices.append(i)
        target_indices.append(i + 1)
        values.append(current_count)
        current_count = stage_count

    labels.append(f"Final Cohort (n={current_count})")

    fig = go.Figure(data=go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="#2a9d8f"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Count: %{value}<extra></extra>'
        )
    ))

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig