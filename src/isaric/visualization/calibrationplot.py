import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import Optional


class CalibrationPlot:
    """
    Calibration plots for assessing agreement between predicted and observed outcomes.
    Works for: survival models, logistic regression, risk prediction models, etc.

    The Plotly methods are interactive and meant for on-screen inspection.
    `save_png` renders the same aggregated curve with matplotlib, for report
    artifacts: Plotly static export would require `kaleido`, an extra
    dependency the package does not declare (NFR007).
    """

    @staticmethod
    def save_png(
        calibration_df: pd.DataFrame,
        title: str = 'Calibration Plot',
        output_path: str = 'calibration_plot.png',
    ) -> str:
        """
        Save a calibration curve as a PNG, from an already aggregated table.

        Args:
            calibration_df: DataFrame with 'predicted' and 'observed' columns,
                one row per bin — as returned by
                `modelevaluation.calibration.compute_calibration_curve`.
            title: Plot title.
            output_path: Destination path for the PNG artifact.

        Returns:
            str: Path to the saved file.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle='--', color='0.5', linewidth=1,
                label='Calibração perfeita')
        ax.plot(calibration_df['predicted'], calibration_df['observed'],
                marker='o', markersize=5, linewidth=2, color='#1f77b4',
                label='Modelo')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Probabilidade predita')
        ax.set_ylabel('Frequência observada')
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(loc='upper left', frameon=False)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    @staticmethod
    def plot(
        predicted: np.ndarray,
        observed: np.ndarray,
        title: str = 'Calibration Plot',
        xlabel: str = 'Predicted Probability',
        ylabel: str = 'Observed Probability',
        height: int = 600,
        width: int = 900,
        show_perfect: bool = True,
    ) -> go.Figure:
        """
        Create a calibration plot.

        Args:
            predicted: Array of predicted probabilities/values.
            observed: Array of observed probabilities/values.
            show_perfect: Show perfect calibration diagonal line.
        """
        fig = go.Figure()

        if show_perfect:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines', name='Perfect Calibration',
                line=dict(color='gray', dash='dash', width=2),
                hoverinfo='skip',
            ))

        fig.add_trace(go.Scatter(
            x=predicted, y=observed,
            mode='lines+markers', name='Model Calibration',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title=ylabel,
            height=height, width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest', showlegend=True,
            legend=dict(x=0.6, y=0.1),
        )
        return fig

    @staticmethod
    def binned_calibration(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
        strategy: str = 'quantile',
        title: str = 'Calibration Plot',
        height: int = 600,
        width: int = 600,
    ) -> go.Figure:
        """
        Binned calibration plot using sklearn's calibration_curve.

        Args:
            y_true: True binary outcomes (0/1).
            y_pred: Predicted probabilities.
            strategy: Binning strategy ('uniform' or 'quantile').
        """
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy=strategy
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2),
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=mean_predicted_value, y=fraction_of_positives,
            mode='lines+markers', name='Model Calibration',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate='Mean Predicted: %{x:.3f}<br>Fraction Positive: %{y:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            height=height, width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest', showlegend=True,
            legend=dict(x=0.6, y=0.1),
        )
        return fig

    @staticmethod
    def survival_calibration(
        fitted_model,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        t: float,
        n_bins: int = 10,
        title: str = 'Survival Calibration Plot',
        height: int = 600,
        width: int = 800,
    ) -> go.Figure:
        """
        Calibration plot for survival models: predicted probability vs KM-observed.

        Args:
            fitted_model: Fitted CoxPH model (lifelines or similar).
            t: Time point for evaluation.
        """
        from lifelines import KaplanMeierFitter

        predicted_survival = fitted_model.predict_survival_function(df, times=[t]).squeeze()

        calib_df = pd.DataFrame({
            'pred': predicted_survival,
            'duration': df[duration_col],
            'event': df[event_col],
        })
        calib_df['bin'] = pd.qcut(calib_df['pred'], n_bins, labels=False, duplicates='drop')

        observed_probs = []
        predicted_probs = []
        kmf = KaplanMeierFitter()

        for i in sorted(calib_df['bin'].unique()):
            bin_df = calib_df[calib_df['bin'] == i]
            predicted_probs.append(bin_df['pred'].mean())
            kmf.fit(bin_df['duration'], event_observed=bin_df['event'])
            observed_probs.append(kmf.predict(t))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines', name='Perfect Calibration',
            line=dict(color='black', dash='dash'),
            hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=predicted_probs, y=observed_probs,
            mode='lines+markers', name='Model Performance',
            marker=dict(size=10, symbol='circle'),
            line=dict(color='blue', width=2),
            hovertemplate='Predicted: %{x:.3f}<br>Observed (KM): %{y:.3f}<extra></extra>',
        ))

        fig.update_layout(
            title=f"{title} (at time t={t})",
            xaxis_title='Predicted Survival Probability',
            yaxis_title='Observed Survival Fraction (Kaplan-Meier)',
            xaxis=dict(range=[0, 1], constrain='domain'),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            width=width, height=height,
            template='plotly_white',
        )
        return fig
