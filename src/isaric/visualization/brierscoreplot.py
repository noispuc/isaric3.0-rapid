import plotly.graph_objs as go
import numpy as np
from typing import Optional


class BrierScorePlot:
    """
    Plotting utilities for Time-Dependent Brier Score in survival analysis.
    """

    @staticmethod
    def brier_score(
        time_points: np.ndarray,
        brier_scores: np.ndarray,
        target_time: Optional[float] = None,
        title: str = 'Time-Dependent Brier Score (IPCW)',
        height: int = 600,
        width: int = 900,
    ) -> go.Figure:
        """
        Create an interactive Brier Score plot over time.

        Args:
            time_points: Evaluation time points.
            brier_scores: Calculated Brier scores at each time point.
            target_time: Specific time point to highlight with a marker.
            title: Plot title.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_points, y=brier_scores,
            mode='lines', name='Brier Score',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Time: %{x:.2f}<br>Score: %{y:.4f}<extra></extra>',
        ))

        fig.add_hline(
            y=0.25, line_dash='dash', line_color='red',
            annotation_text='Non-informative (BS=0.25)',
            annotation_position='bottom right',
        )

        if target_time is not None:
            idx = np.argmin(np.abs(time_points - target_time))
            score_at_t = brier_scores[idx]
            fig.add_trace(go.Scatter(
                x=[time_points[idx]], y=[score_at_t],
                mode='markers',
                marker=dict(color='black', size=12, symbol='x'),
                name=f'Score at t={target_time}: {score_at_t:.3f}',
                hovertemplate='Target Time: %{x:.2f}<br>Score: %{y:.4f}<extra></extra>',
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Follow-up Time',
            yaxis_title='Brier Score',
            yaxis=dict(range=[0, max(0.3, max(brier_scores) * 1.2)]),
            template='plotly_white',
            height=height, width=width,
            hovermode='x unified',
        )
        return fig
