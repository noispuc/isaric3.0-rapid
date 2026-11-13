"""
Sankey diagram visualization for the RAPID methodology.

This module provides functions to generate Sankey diagrams (Step 6.6
of the RAPID methodology). Sankey diagrams visualize the flow of
quantities from one set of categories to another.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- patient_pathway: Visualize patient transitions between states.
- cohort_flow: Display inclusion/exclusion criteria flow.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


def patient_pathway(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    value_col: Optional[str] = None,
    title: str = "Patient Pathway",
    height: int = 600,
    width: int = 900,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [source_col, target_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _patient_pathway_plotly(
            data=data,
            source_col=source_col,
            target_col=target_col,
            value_col=value_col,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _patient_pathway_matplotlib(
            data=data,
            source_col=source_col,
            target_col=target_col,
            value_col=value_col,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def cohort_flow(
    data: pd.DataFrame,
    stages: List[str],
    title: str = "Cohort Flow",
    height: int = 600,
    width: int = 900,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If stages are not found or backend invalid.
    """
    for col in stages:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _cohort_flow_plotly(
            data=data,
            stages=stages,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _cohort_flow_matplotlib(
            data=data,
            stages=stages,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _patient_pathway_plotly(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    value_col: Optional[str],
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly patient pathway Sankey diagram."""
    # Count transitions if value_col is None
    if value_col is None:
        flows = data.groupby([source_col, target_col]).size().reset_index(name='count')
        value_col = 'count'
    else:
        flows = data.copy()

    all_states = list(set(
        flows[source_col].unique().tolist() +
        flows[target_col].unique().tolist()
    ))

    state_to_idx = {state: i for i, state in enumerate(all_states)}

    source_indices = flows[source_col].map(state_to_idx).tolist()
    target_indices = flows[target_col].map(state_to_idx).tolist()
    values = flows[value_col].tolist()

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


def _cohort_flow_plotly(
    data: pd.DataFrame,
    stages: List[str],
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly cohort flow Sankey diagram."""
    labels = []
    source_indices = []
    target_indices = []
    values = []

    current_count = len(data)
    labels.append("Initial Cohort")

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


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _patient_pathway_matplotlib(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    value_col: Optional[str],
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib patient pathway Sankey diagram (simplified)."""
    if value_col is None:
        flows = data.groupby([source_col, target_col]).size().reset_index(name='count')
        value_col = 'count'
    else:
        flows = data.copy()

    all_states = list(set(
        flows[source_col].unique().tolist() +
        flows[target_col].unique().tolist()
    ))

    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Simplified: Bar chart showing flows between states
    state_counts = flows.groupby(source_col)[value_col].sum().sort_values(ascending=False)
    
    bars = ax.bar(range(len(state_counts)), state_counts.values, color='#2a9d8f')
    ax.set_xticks(range(len(state_counts)))
    ax.set_xticklabels(state_counts.index.tolist(), rotation=45, ha='right')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add flow information as text
    for i, (_, row) in enumerate(flows.iterrows()):
        ax.text(
            i % len(state_counts),
            state_counts.max() * 0.9,
            f'{row[source_col]} → {row[target_col]}: {row[value_col]}',
            fontsize=8,
            ha='center'
        )
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _cohort_flow_matplotlib(
    data: pd.DataFrame,
    stages: List[str],
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib cohort flow diagram."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    current_count = len(data)
    labels = ["Initial"]
    counts = [current_count]
    
    for stage in stages:
        stage_count = int(data[stage].sum())
        labels.append(stage)
        counts.append(stage_count)
        current_count = stage_count
    
    labels.append("Final")
    counts.append(current_count)
    
    # Draw rectangles
    box_width = 0.6
    box_height = 0.6
    x_positions = range(len(labels))
    
    for i, (label, count) in enumerate(zip(labels, counts)):
        ax.add_patch(Rectangle(
            (i - box_width/2, 0),
            box_width,
            box_height,
            fill=True,
            facecolor='#2a9d8f',
            edgecolor='black',
            alpha=0.7
        ))
        ax.text(i, box_height/2, f'{label}\n(n={count})', ha='center', va='center', fontsize=10)
    
    # Draw arrows
    for i in range(len(labels) - 1):
        ax.annotate(
            '',
            xy=(i + 1 - box_width/2, box_height/2),
            xytext=(i + box_width/2, box_height/2),
            arrowprops=dict(arrowstyle='->', color='black', lw=2)
        )
    
    ax.set_xlim(-1, len(labels))
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')
    ax.set_title(title)
    fig.tight_layout()
    return fig