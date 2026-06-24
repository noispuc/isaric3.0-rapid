import plotly.graph_objs as go
import pandas as pd
import numpy as np
from typing import Optional


class CalibrationPlot:
    """
    Calibration plots for assessing agreement between predicted and observed outcomes.
    Works for: survival models, logistic regression, risk prediction models, etc.
    """

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
