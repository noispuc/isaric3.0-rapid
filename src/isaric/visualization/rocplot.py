import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from typing import Optional, Dict, List, Union


class ROCPlot:
    """
    ROC curve plotting for classification and risk prediction models.
    """

    @staticmethod
    def plot(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc: float,
        title: str = 'ROC Curve',
        label: Optional[str] = None,
        height: int = 600,
        width: int = 600,
        show_diagonal: bool = True,
    ) -> go.Figure:
        """
        Create an interactive ROC curve plot.

        Args:
            fpr: False positive rates.
            tpr: True positive rates.
            auc: Area under the curve value.
            label: Legend label (defaults to AUC value).
            show_diagonal: Show random chance diagonal.
        """
        if label is None:
            label = f'ROC Curve (AUC = {auc:.3f})'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines', name=label,
            line=dict(color='blue', width=3),
            fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
        ))

        if show_diagonal:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Random Chance (AUC = 0.5)',
                line=dict(color='gray', dash='dash', width=2),
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            height=height, width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest', showlegend=True,
            legend=dict(x=0.6, y=0.1),
        )
        return fig

    @staticmethod
    def compare_multiple(
        roc_data: List[Dict[str, Union[np.ndarray, float, str]]],
        title: str = 'ROC Curve Comparison',
        height: int = 600,
        width: int = 600,
    ) -> go.Figure:
        """
        Plot multiple ROC curves for comparison.

        Args:
            roc_data: List of dicts with keys 'fpr', 'tpr', 'auc', 'label'.
        """
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, data in enumerate(roc_data):
            label = f"{data['label']} (AUC = {data['auc']:.3f})"
            fig.add_trace(go.Scatter(
                x=data['fpr'], y=data['tpr'],
                mode='lines', name=label,
                line=dict(color=colors[i % len(colors)], width=3),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Random Chance',
            line=dict(color='gray', dash='dash', width=2),
            hoverinfo='skip',
        ))

        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            height=height, width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest', showlegend=True,
            legend=dict(x=0.6, y=0.1),
        )
        return fig
