import plotly.graph_objs as go
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Optional


class ResidualPlots:
    """
    Residual plots for regression diagnostics.
    Works for: linear regression, GLMs, survival models, etc.
    """

    @staticmethod
    def residuals_vs_fitted(
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        title: str = 'Residuals vs Fitted',
        xlabel: str = 'Fitted Values',
        ylabel: str = 'Residuals',
        add_smoother: bool = False,
        height: int = 600,
        width: int = 900,
    ) -> go.Figure:
        """
        Plot residuals against fitted values (classic regression diagnostic).

        Args:
            residuals: Array of residual values.
            fitted_values: Array of fitted/predicted values.
            add_smoother: Add LOWESS smoother line.
        """
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fitted_values, y=residuals,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Residuals',
            hovertemplate='Fitted: %{x}<br>Residual: %{y}<extra></extra>',
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2,
                      annotation_text='y=0', annotation_position='right')

        if add_smoother:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted_values, frac=0.3)
            fig.add_trace(go.Scatter(
                x=smoothed[:, 0], y=smoothed[:, 1],
                mode='lines',
                line=dict(color='orange', width=3),
                name='LOWESS Smoother',
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=title, xaxis_title=xlabel, yaxis_title=ylabel,
            height=height, width=width, hovermode='closest', showlegend=True,
        )
        return fig

    @staticmethod
    def residuals_vs_covariate(
        residuals: np.ndarray,
        covariate: np.ndarray,
        covariate_name: str,
        residual_type: str = 'Residuals',
        add_smoother: bool = True,
        height: int = 600,
        width: int = 800,
    ) -> go.Figure:
        """
        Plot residuals against a covariate. Uses box plot for categorical, scatter for continuous.
        """
        plot_df = pd.DataFrame({'residuals': residuals, 'covariate': covariate}).dropna()

        is_categorical = (
            pd.api.types.is_categorical_dtype(plot_df['covariate'])
            or pd.api.types.is_object_dtype(plot_df['covariate'])
            or plot_df['covariate'].dtype == bool
            or len(plot_df['covariate'].unique()) < 10
        )

        fig = go.Figure()

        if is_categorical:
            for cat in sorted(plot_df['covariate'].unique()):
                fig.add_trace(go.Box(
                    y=plot_df[plot_df['covariate'] == cat]['residuals'],
                    name=str(cat), boxmean='sd',
                ))
            fig.update_layout(xaxis_title=covariate_name, yaxis_title=residual_type, showlegend=False)
        else:
            fig.add_trace(go.Scatter(
                x=plot_df['covariate'], y=plot_df['residuals'],
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Residuals',
                hovertemplate=f'{covariate_name}: %{{x:.3f}}<br>{residual_type}: %{{y:.3f}}<extra></extra>',
            ))
            if add_smoother:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(plot_df['residuals'], plot_df['covariate'], frac=0.3)
                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0], y=smoothed[:, 1],
                    mode='lines', line=dict(color='orange', width=3),
                    name='LOWESS Smoother', hoverinfo='skip',
                ))
            fig.update_layout(xaxis_title=covariate_name, yaxis_title=residual_type, showlegend=True)

        fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
        fig.update_layout(
            title=f'{residual_type} vs {covariate_name}',
            height=height, width=width, hovermode='closest',
        )
        return fig

    @staticmethod
    def qq_plot(
        residuals: np.ndarray,
        title: str = 'Q-Q Plot',
        height: int = 600,
        width: int = 600,
    ) -> go.Figure:
        """Q-Q plot for assessing normality of residuals."""
        qq = stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq[0][0]
        ordered_residuals = qq[0][1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles, y=ordered_residuals,
            mode='markers', marker=dict(color='blue', size=8),
            name='Sample Quantiles',
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles, y=theoretical_quantiles,
            mode='lines', line=dict(color='red', dash='dash', width=2),
            name='Ideal Normal', hoverinfo='skip',
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles',
            height=height, width=width, hovermode='closest', showlegend=True,
        )
        return fig

    @staticmethod
    def deviance_residuals(
        fitted_model,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        covariate_name: str,
        height: int = 600,
        width: int = 800,
    ) -> go.Figure:
        """Deviance residuals for a Cox PH model plotted against a covariate."""
        e_observed = df[event_col].values
        risk_scores = fitted_model.predict_partial_hazard(df).values

        baseline_hazard_df = fitted_model.baseline_cumulative_hazard_
        times = baseline_hazard_df.index.values
        hazard_values = baseline_hazard_df.iloc[:, 0].values

        indices = np.searchsorted(times, df[duration_col].values, side='right') - 1
        indices = np.maximum(0, indices)
        cum_baseline_hazard = hazard_values[indices]

        cumulative_subject_hazard = cum_baseline_hazard * risk_scores
        martingale_residuals = e_observed - cumulative_subject_hazard

        log_component = np.zeros_like(martingale_residuals)
        diff = e_observed - martingale_residuals
        mask = diff > 0
        log_component[mask] = np.log(diff[mask])

        term = martingale_residuals + e_observed * log_component
        deviance_res = np.sign(martingale_residuals) * np.sqrt(2 * -term)

        fig = ResidualPlots.residuals_vs_covariate(
            residuals=deviance_res,
            covariate=df[covariate_name].values,
            covariate_name=covariate_name,
            residual_type='Deviance Residuals',
            height=height, width=width,
        )
        fig.update_layout(title=f'Cox Model Diagnostics: Deviance Residuals vs {covariate_name}')
        return fig

    @staticmethod
    def schoenfeld_plot(
        times: np.ndarray,
        residuals: np.ndarray,
        covariate_name: str,
        height: int = 600,
        width: int = 900,
    ) -> go.Figure:
        """Schoenfeld residual plot for checking proportional hazards assumption."""
        import statsmodels.api as sm

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=residuals,
            mode='markers',
            marker=dict(color='#1f77b4', size=6, opacity=0.5),
            name='Schoenfeld Residuals',
            hovertemplate='Time: %{x}<br>Residual: %{y:.4f}<extra></extra>',
        ))

        smoothed = sm.nonparametric.lowess(residuals, times, frac=0.3)
        fig.add_trace(go.Scatter(
            x=smoothed[:, 0], y=smoothed[:, 1],
            mode='lines', line=dict(color='#d62728', width=3),
            name='LOWESS Smoother',
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
        fig.update_layout(
            title=f'Schoenfeld Residuals: {covariate_name} (PH Assumption Check)',
            xaxis_title='Time',
            yaxis_title=f'Residuals for {covariate_name}',
            template='plotly_white',
            height=height, width=width, hovermode='x',
        )
        return fig
