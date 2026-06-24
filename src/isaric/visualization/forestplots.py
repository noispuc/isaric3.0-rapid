import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import Optional, Dict


class ForestPlot:
    """
    General-purpose forest plot for visualizing effect sizes with confidence intervals.
    Works for: Hazard Ratios, Odds Ratios, Risk Ratios, Coefficients, etc.
    """

    @staticmethod
    def plot(
        df: pd.DataFrame,
        effect_col: str,
        lower_col: str,
        upper_col: str,
        label_col: str,
        title: str = 'Forest Plot',
        xaxis_title: str = 'Effect Size',
        null_value: float = 1.0,
        sort: bool = True,
        ascending: bool = True,
        marker_color: str = 'blue',
        marker_size: int = 10,
        line_color: str = 'blue',
        line_width: int = 2,
        null_line_color: str = 'red',
        null_line_width: int = 2,
        height: int = 600,
        margin: Optional[Dict[str, int]] = None,
        show_values: bool = False,
        log_scale: bool = False,
    ) -> go.Figure:
        """
        Create an interactive forest plot using Plotly.

        Args:
            df: DataFrame with effect sizes and confidence intervals.
            effect_col: Column for effect size (HR, OR, coef, etc.).
            lower_col: Column for lower CI bound.
            upper_col: Column for upper CI bound.
            label_col: Column for row labels.
            title: Plot title.
            xaxis_title: X-axis label.
            null_value: Reference line value (1.0 for ratios, 0.0 for differences).
            sort: Whether to sort by effect size.
            ascending: Sort direction.
            marker_color: Color for point estimates.
            marker_size: Size of point markers.
            line_color: Color for CI lines.
            line_width: Width of CI lines.
            null_line_color: Color for null reference line.
            null_line_width: Width of null line.
            height: Plot height in pixels.
            margin: Custom margins dict.
            show_values: Display values as annotations.
            log_scale: Use log scale for x-axis.

        Returns:
            Plotly Figure object.
        """
        plot_df = df.copy()

        if sort:
            plot_df = plot_df.sort_values(by=effect_col, ascending=ascending)

        if margin is None:
            margin = dict(l=200, r=100, t=100, b=50)

        traces = []

        traces.append(go.Scatter(
            x=plot_df[effect_col],
            y=plot_df[label_col],
            mode='markers',
            name=xaxis_title,
            marker=dict(color=marker_color, size=marker_size),
            hovertemplate='%{y}<br>%{x:.3f}<extra></extra>',
        ))

        for _, row in plot_df.iterrows():
            traces.append(go.Scatter(
                x=[row[lower_col], row[upper_col]],
                y=[row[label_col], row[label_col]],
                mode='lines',
                showlegend=False,
                line=dict(color=line_color, width=line_width),
                hoverinfo='skip',
            ))

        annotations = []
        if show_values:
            for _, row in plot_df.iterrows():
                annotations.append(dict(
                    x=row[effect_col],
                    y=row[label_col],
                    text=f"{row[effect_col]:.2f}<br>({row[lower_col]:.2f}-{row[upper_col]:.2f})",
                    showarrow=False,
                    xshift=50,
                    font=dict(size=9),
                ))

        xaxis_config = dict(title=xaxis_title)
        if log_scale:
            xaxis_config['type'] = 'log'

        layout = go.Layout(
            title=title,
            xaxis=xaxis_config,
            yaxis=dict(
                title='',
                automargin=True,
                tickmode='array',
                tickvals=plot_df[label_col].tolist(),
                ticktext=plot_df[label_col].tolist(),
            ),
            shapes=[dict(
                type='line',
                x0=null_value, y0=-0.5,
                x1=null_value, y1=len(plot_df) - 0.5,
                line=dict(color=null_line_color, width=null_line_width, dash='dash'),
            )],
            margin=margin,
            height=height,
            annotations=annotations,
        )

        return go.Figure(data=traces, layout=layout)


# ------------------------------------------------------------------
# Compatibility wrapper
# ------------------------------------------------------------------

def plot_forest(coefficients, ci_lower, ci_upper, labels, title="Forest Plot"):
    """
    Backwards-compatible wrapper. Builds a minimal DataFrame and delegates to ForestPlot.plot().
    """
    df = pd.DataFrame({
        'label': labels,
        'effect': coefficients,
        'lower': ci_lower,
        'upper': ci_upper,
    })
    return ForestPlot.plot(
        df=df,
        effect_col='effect',
        lower_col='lower',
        upper_col='upper',
        label_col='label',
        title=title,
        null_value=0.0,
        sort=False,
    )
