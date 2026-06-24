import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd


class LCAPlots:
    """
    Plotting utilities for Latent Class Analysis (LCA).
    Provides interactive Plotly versions of all LCA diagnostic and profile plots.
    """

    @staticmethod
    def plot_profiles(prob_df: pd.DataFrame, n_components: int) -> go.Figure:
        """
        Heatmap of conditional probabilities per feature and latent class.

        Args:
            prob_df: DataFrame where rows are classes and columns are variables.
            n_components: Number of latent classes.
        """
        y_labels = [f'Class {i}' for i in range(n_components)]

        fig = px.imshow(
            prob_df,
            labels=dict(x="Clinical Variables", y="Latent Class", color="Probability"),
            x=prob_df.columns,
            y=y_labels,
            color_continuous_scale='Viridis',
            text_auto='.2f',
            aspect='auto',
        )
        fig.update_layout(
            title=f'Phenotype Profiles (K={n_components})',
            xaxis_tickangle=-45,
            template='plotly_white',
            coloraxis_colorbar=dict(title="Prob"),
        )
        return fig

    @staticmethod
    def plot_clusters(clusters_series: pd.Series) -> go.Figure:
        """
        Bar chart showing absolute and relative frequency of each latent class.

        Args:
            clusters_series: Series containing class assignment for each row.
        """
        counts = clusters_series.value_counts().sort_index()
        total = counts.sum()
        percentages = (counts / total * 100).round(2)
        labels = [f'Class {i}' for i in counts.index]

        fig = go.Figure(data=[go.Bar(
            x=labels, y=counts.values,
            text=[f'{p}%' for p in percentages],
            textposition='auto',
            marker_color='rgb(55, 83, 109)',
        )])
        fig.update_layout(
            title='Latent Class Distribution',
            xaxis_title='Identified Phenotypes',
            yaxis_title='Number of Patients',
            template='plotly_white',
        )
        return fig

    @staticmethod
    def plot_model_selection(results_df: pd.DataFrame) -> go.Figure:
        """
        Elbow plot comparing AIC and BIC across different K values.

        Args:
            results_df: DataFrame with columns ['n_clusters', 'AIC', 'BIC'].
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['n_clusters'], y=results_df['AIC'],
            mode='lines+markers', name='AIC',
            line=dict(shape='linear', color='royalblue'),
        ))
        fig.add_trace(go.Scatter(
            x=results_df['n_clusters'], y=results_df['BIC'],
            mode='lines+markers', name='BIC',
            line=dict(shape='linear', color='firebrick'),
        ))
        fig.update_layout(
            title='Model Selection Criteria (Information Theory)',
            xaxis_title='Number of Latent Classes (K)',
            yaxis_title='Metric Value (Lower is better)',
            legend_title='Metrics',
            template='plotly_white',
            hovermode='x unified',
        )
        return fig

    @staticmethod
    def plot_grid_metrics(
        grid_results_df: pd.DataFrame,
        metrics: list = ['LL', 'AIC', 'BIC', 'CAIC', 'SABIC'],
    ) -> go.Figure:
        """
        Multiple metrics across K values during grid search.

        Args:
            grid_results_df: DataFrame with 'n_clusters' and metric columns.
            metrics: List of metric column names to plot.
        """
        fig = go.Figure()
        for m in metrics:
            if m in grid_results_df.columns:
                fig.add_trace(go.Scatter(
                    x=grid_results_df['n_clusters'], y=grid_results_df[m],
                    mode='lines+markers', name=m,
                ))
        fig.update_layout(
            title='Grid Search: Information Criteria & Log-Likelihood',
            xaxis_title='Number of Latent Classes (K)',
            yaxis_title='Metric Value',
            template='plotly_white',
            hovermode='x unified',
        )
        return fig

    @staticmethod
    def plot_grid_entropy(grid_results_df: pd.DataFrame) -> go.Figure:
        """
        Absolute and relative entropy metrics across K values.

        Args:
            grid_results_df: DataFrame with 'n_clusters', 'entropy', 'relative_entropy' columns.
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if 'entropy' in grid_results_df.columns:
            fig.add_trace(go.Scatter(
                x=grid_results_df['n_clusters'], y=grid_results_df['entropy'],
                mode='lines+markers', name='Entropy',
                line=dict(color='orange'),
            ), secondary_y=False)

        if 'relative_entropy' in grid_results_df.columns:
            fig.add_trace(go.Scatter(
                x=grid_results_df['n_clusters'], y=grid_results_df['relative_entropy'],
                mode='lines+markers', name='Relative Entropy',
                line=dict(color='green'),
            ), secondary_y=True)

        fig.update_layout(
            title='Grid Search: Entropy Metrics',
            xaxis_title='Number of Latent Classes (K)',
            template='plotly_white',
            hovermode='x unified',
        )
        fig.update_yaxes(title_text='Entropy', secondary_y=False)
        fig.update_yaxes(title_text='Relative Entropy', secondary_y=True)
        return fig

    @staticmethod
    def plot_conditional_probs_line(prob_df: pd.DataFrame) -> go.Figure:
        """
        Line plot of conditional probabilities per feature across classes.

        Args:
            prob_df: DataFrame where rows are classes and columns are features.
        """
        fig = go.Figure()
        variables = prob_df.columns.tolist()

        for i in range(len(prob_df)):
            fig.add_trace(go.Scatter(
                x=variables, y=prob_df.iloc[i].tolist(),
                mode='lines+markers', name=f'Class {i}',
            ))

        fig.update_layout(
            title='Conditional Probabilities by Feature',
            xaxis_title='Features', yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            xaxis_tickangle=-45,
            template='plotly_white',
            hovermode='x unified',
        )
        return fig

    @staticmethod
    def plot_radar_profiles(prob_df: pd.DataFrame) -> go.Figure:
        """
        Radar charts per latent class showing conditional probability profiles.
        Replaces the matplotlib plot_radar_per_class() in modeling/LCA.py.

        Args:
            prob_df: DataFrame where rows are classes and columns are features.
        """
        fig = go.Figure()
        categories = prob_df.columns.tolist()
        categories_closed = categories + [categories[0]]

        for i in range(len(prob_df)):
            values = prob_df.iloc[i].tolist()
            values_closed = values + [values[0]]
            fig.add_trace(go.Scatterpolar(
                r=values_closed, theta=categories_closed,
                fill='toself', name=f'Class {i}', opacity=0.3,
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Phenotype Radial Profiles',
            template='plotly_white',
        )
        return fig
