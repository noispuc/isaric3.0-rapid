import plotly.graph_objs as go
import numpy as np
from typing import Optional, List


class ConfusionMatrixPlot:
    """
    Confusion matrix visualization for classification models.
    """

    @staticmethod
    def plot(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        normalize: bool = False,
        show_values: bool = True,
        colorscale: str = 'Blues',
        height: int = 600,
        width: int = 600,
    ) -> go.Figure:
        """
        Create a confusion matrix heatmap.

        Args:
            confusion_matrix: 2D array of confusion matrix values.
            class_names: List of class names (defaults to indices).
            normalize: If True, normalize by row (true label).
            show_values: Display values in cells.
            colorscale: Plotly colorscale name.
        """
        cm = confusion_matrix.copy()

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        text = []
        for i in range(len(cm)):
            row_text = []
            for j in range(len(cm[i])):
                row_text.append(f'{cm[i][j]:.1%}' if normalize else f'{int(cm[i][j])}')
            text.append(row_text)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names, y=class_names,
            text=text if show_values else None,
            texttemplate='%{text}' if show_values else None,
            textfont=dict(size=14),
            colorscale=colorscale, showscale=True,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted', yaxis_title='True',
            height=height, width=width,
            yaxis=dict(autorange='reversed'),
        )
        return fig

    @staticmethod
    def plot_with_metrics(
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = 'Confusion Matrix with Metrics',
        normalize: bool = False,
        colorscale: str = 'Blues',
        height: int = 700,
        width: int = 900,
    ) -> go.Figure:
        """
        Confusion matrix with accuracy, precision, recall, and F1 as annotations.
        """
        cm = confusion_matrix.copy()

        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]

        if len(cm) == 2:
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics_text = (
                f"<b>Performance Metrics:</b><br>"
                f"Accuracy: {accuracy:.3f}<br>"
                f"Precision: {precision:.3f}<br>"
                f"Recall: {recall:.3f}<br>"
                f"F1 Score: {f1:.3f}"
            )
        else:
            accuracy = np.trace(cm) / np.sum(cm)
            metrics_text = f"<b>Overall Accuracy:</b> {accuracy:.3f}"

        fig = ConfusionMatrixPlot.plot(
            confusion_matrix=cm,
            class_names=class_names,
            title=title,
            normalize=normalize,
            show_values=True,
            colorscale=colorscale,
            height=height, width=width,
        )

        fig.add_annotation(
            text=metrics_text,
            xref='paper', yref='paper',
            x=1.15, y=0.5,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black', borderwidth=1, borderpad=10,
        )
        return fig
