"""
Modular plotting utilities for statistical analysis using Plotly.
Provides reusable plot types for survival analysis, regression, and other statistical methods.
All plots are interactive and publication-ready.
"""

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Optional, Dict, List, Union, Tuple


# ============================================================================
# 1. FOREST PLOTS (Effect Size Visualization)
# ============================================================================

class ForestPlot:
    """
    General-purpose forest plot for visualizing effect sizes with confidence intervals.
    Works for: Hazard Ratios, Odds Ratios, Risk Ratios, Coefficients, etc.
    """
    
    @staticmethod
    def plot(df: pd.DataFrame,
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
             log_scale: bool = False) -> go.Figure:
        """
        Create a forest plot using Plotly.
        
        Args:
            df: DataFrame with effect sizes and confidence intervals
            effect_col: Column name for effect size (HR, OR, coef, etc.)
            lower_col: Column name for lower CI bound
            upper_col: Column name for upper CI bound
            label_col: Column name for row labels
            title: Plot title
            xaxis_title: X-axis label
            null_value: Null/reference line value (1.0 for ratios, 0.0 for differences)
            sort: Whether to sort by effect size
            ascending: Sort direction
            marker_color: Color for point estimates
            marker_size: Size of point markers
            line_color: Color for CI lines
            line_width: Width of CI lines
            null_line_color: Color for null reference line
            null_line_width: Width of null line
            height: Plot height in pixels
            margin: Custom margins dict
            show_values: Display values as annotations
            log_scale: Use log scale for x-axis
            
        Returns:
            Plotly Figure object
        """
        plot_df = df.copy()
        
        if sort:
            plot_df = plot_df.sort_values(by=effect_col, ascending=ascending)
        
        if margin is None:
            margin = dict(l=200, r=100, t=100, b=50)
        
        traces = []
        
        # Point estimates
        traces.append(
            go.Scatter(
                x=plot_df[effect_col],
                y=plot_df[label_col],
                mode='markers',
                name=xaxis_title,
                marker=dict(color=marker_color, size=marker_size),
                hovertemplate='%{y}<br>%{x:.3f}<extra></extra>'
            )
        )
        
        # Confidence intervals
        for _, row in plot_df.iterrows():
            traces.append(
                go.Scatter(
                    x=[row[lower_col], row[upper_col]],
                    y=[row[label_col], row[label_col]],
                    mode='lines',
                    showlegend=False,
                    line=dict(color=line_color, width=line_width),
                    hoverinfo='skip'
                )
            )
        
        # Annotations for values
        annotations = []
        if show_values:
            for _, row in plot_df.iterrows():
                annotations.append(
                    dict(
                        x=row[effect_col],
                        y=row[label_col],
                        text=f"{row[effect_col]:.2f}<br>({row[lower_col]:.2f}-{row[upper_col]:.2f})",
                        showarrow=False,
                        xshift=50,
                        font=dict(size=9)
                    )
                )
        
        # Layout
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
                ticktext=plot_df[label_col].tolist()
            ),
            shapes=[
                dict(
                    type='line',
                    x0=null_value,
                    y0=-0.5,
                    x1=null_value,
                    y1=len(plot_df) - 0.5,
                    line=dict(color=null_line_color, width=null_line_width, dash='dash')
                )
            ],
            margin=margin,
            height=height,
            annotations=annotations
        )
        
        fig = go.Figure(data=traces, layout=layout)
        return fig


# ============================================================================
# 2. RESIDUAL PLOTS (Diagnostic Plots)
# ============================================================================

class ResidualPlots:
    """
    General-purpose residual plotting for regression diagnostics.
    Works for: Linear regression, GLMs, survival models, etc.
    """
    
    @staticmethod
    def residuals_vs_fitted(residuals: np.ndarray,
                           fitted_values: np.ndarray,
                           title: str = 'Residuals vs Fitted',
                           xlabel: str = 'Fitted Values',
                           ylabel: str = 'Residuals',
                           add_smoother: bool = False,
                           height: int = 600,
                           width: int = 900) -> go.Figure:
        """
        Plot residuals against fitted values (classic regression diagnostic).
        
        Args:
            residuals: Array of residual values
            fitted_values: Array of fitted/predicted values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            add_smoother: Add LOWESS smoother line
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Residuals',
            hovertemplate='Fitted: %{x}<br>Residual: %{y}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2,
                     annotation_text='y=0', annotation_position='right')
        
        # LOWESS smoother
        if add_smoother:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, fitted_values, frac=0.3)
            fig.add_trace(go.Scatter(
                x=smoothed[:, 0],
                y=smoothed[:, 1],
                mode='lines',
                line=dict(color='orange', width=3),
                name='LOWESS Smoother',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=height,
            width=width,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def residuals_vs_covariate(residuals: np.ndarray,
                               covariate: np.ndarray,
                               covariate_name: str,
                               residual_type: str = 'Residuals',
                               add_smoother: bool = True,
                               height: int = 600,
                               width: int = 800) -> go.Figure:
        """
        Plot residuals against a specific covariate.
        Automatically detects categorical vs continuous and uses appropriate plot type.
        
        Args:
            residuals: Array of residual values
            covariate: Array of covariate values
            covariate_name: Name of the covariate (for labels)
            residual_type: Type of residuals (for y-axis label)
            add_smoother: Add LOWESS smoother for continuous variables
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        plot_df = pd.DataFrame({
            'residuals': residuals,
            'covariate': covariate
        }).dropna()
        
        # Detect if categorical or continuous
        is_categorical = (
            pd.api.types.is_categorical_dtype(plot_df['covariate']) or
            pd.api.types.is_object_dtype(plot_df['covariate']) or
            plot_df['covariate'].dtype == bool or
            len(plot_df['covariate'].unique()) < 10
        )
        
        fig = go.Figure()
        
        if is_categorical:
            # Box plot for categorical
            categories = sorted(plot_df['covariate'].unique())
            for cat in categories:
                cat_data = plot_df[plot_df['covariate'] == cat]['residuals']
                fig.add_trace(go.Box(
                    y=cat_data,
                    name=str(cat),
                    boxmean='sd'
                ))
            fig.update_layout(
                xaxis_title=covariate_name,
                yaxis_title=residual_type,
                showlegend=False
            )
        else:
            # Scatter plot for continuous
            fig.add_trace(go.Scatter(
                x=plot_df['covariate'],
                y=plot_df['residuals'],
                mode='markers',
                marker=dict(color='blue', size=8, opacity=0.6),
                name='Residuals',
                hovertemplate=f'{covariate_name}: %{{x:.3f}}<br>{residual_type}: %{{y:.3f}}<extra></extra>'
            ))
            
            if add_smoother:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(plot_df['residuals'], plot_df['covariate'], frac=0.3)
                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0],
                    y=smoothed[:, 1],
                    mode='lines',
                    line=dict(color='orange', width=3),
                    name='LOWESS Smoother',
                    hoverinfo='skip'
                ))
            
            fig.update_layout(
                xaxis_title=covariate_name,
                yaxis_title=residual_type,
                showlegend=True
            )
        
        # Zero line
        fig.add_hline(y=0, line_dash='dash', line_color='red', line_width=2)
        
        fig.update_layout(
            title=f'{residual_type} vs {covariate_name}',
            height=height,
            width=width,
            hovermode='closest'
        )
        
        return fig
    
    @staticmethod
    def qq_plot(residuals: np.ndarray,
                title: str = 'Q-Q Plot',
                height: int = 600,
                width: int = 600) -> go.Figure:
        """
        Create a Q-Q plot for assessing normality of residuals.
        
        Args:
            residuals: Array of residual values
            title: Plot title
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        qq = stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq[0][0]
        ordered_residuals = qq[0][1]
        
        fig = go.Figure()
        
        # Sample quantiles
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=ordered_residuals,
            mode='markers',
            marker=dict(color='blue', size=8),
            name='Sample Quantiles',
            hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>'
        ))
        
        # Ideal normal line
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Ideal Normal',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            height=height,
            width=width,
            hovermode='closest',
            showlegend=True
        )
        
        return fig


# ============================================================================
# 3. ROC CURVES (Classification Performance)
# ============================================================================

class ROCPlot:
    """
    ROC curve plotting for classification and risk prediction models.
    Works for: Logistic regression, survival models (time-dependent), classifiers, etc.
    """
    
    @staticmethod
    def plot(fpr: np.ndarray,
             tpr: np.ndarray,
             auc: float,
             title: str = 'ROC Curve',
             label: Optional[str] = None,
             height: int = 600,
             width: int = 600,
             show_diagonal: bool = True) -> go.Figure:
        """
        Create an ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under the curve value
            title: Plot title
            label: Legend label (defaults to AUC value)
            height: Figure height in pixels
            width: Figure width in pixels
            show_diagonal: Show diagonal reference line
            
        Returns:
            Plotly Figure object
        """
        if label is None:
            label = f'ROC Curve (AUC = {auc:.3f})'
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=label,
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Diagonal reference line
        if show_diagonal:
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Chance (AUC = 0.5)',
                line=dict(color='gray', dash='dash', width=2),
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            height=height,
            width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    @staticmethod
    def compare_multiple(roc_data: List[Dict[str, Union[np.ndarray, float, str]]],
                        title: str = 'ROC Curve Comparison',
                        height: int = 600,
                        width: int = 600) -> go.Figure:
        """
        Plot multiple ROC curves for comparison.
        
        Args:
            roc_data: List of dicts with keys 'fpr', 'tpr', 'auc', 'label'
            title: Plot title
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Plotly
        
        for i, data in enumerate(roc_data):
            label = f"{data['label']} (AUC = {data['auc']:.3f})"
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=label,
                line=dict(color=color, width=3),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Chance',
            line=dict(color='gray', dash='dash', width=2),
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            height=height,
            width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig


# ============================================================================
# 4. CONFUSION MATRIX (Classification Performance)
# ============================================================================

class ConfusionMatrixPlot:
    """
    Confusion matrix visualization for classification models.
    Works for: Logistic regression, classifiers, any binary or multiclass classification.
    """
    
    @staticmethod
    def plot(confusion_matrix: np.ndarray,
             class_names: Optional[List[str]] = None,
             title: str = 'Confusion Matrix',
             normalize: bool = False,
             show_values: bool = True,
             colorscale: str = 'Blues',
             height: int = 600,
             width: int = 600) -> go.Figure:
        """
        Create a confusion matrix heatmap.
        
        Args:
            confusion_matrix: 2D array of confusion matrix values
            class_names: List of class names (defaults to indices)
            title: Plot title
            normalize: If True, normalize by row (true label)
            show_values: Display values in cells
            colorscale: Plotly colorscale name
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        cm = confusion_matrix.copy()
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            value_format = '.2%'
        else:
            value_format = 'd'
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create text annotations
        text = []
        for i in range(len(cm)):
            row_text = []
            for j in range(len(cm[i])):
                if normalize:
                    row_text.append(f'{cm[i][j]:.1%}')
                else:
                    row_text.append(f'{int(cm[i][j])}')
            text.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text if show_values else None,
            texttemplate='%{text}' if show_values else None,
            textfont=dict(size=14),
            colorscale=colorscale,
            showscale=True,
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='True',
            height=height,
            width=width,
            yaxis=dict(autorange='reversed')  # Put 0,0 in top-left
        )
        
        return fig
    
    @staticmethod
    def plot_with_metrics(confusion_matrix: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         title: str = 'Confusion Matrix with Metrics',
                         normalize: bool = False,
                         colorscale: str = 'Blues',
                         height: int = 700,
                         width: int = 900) -> go.Figure:
        """
        Create a confusion matrix with accuracy, precision, recall, and F1 metrics displayed.
        
        Args:
            confusion_matrix: 2D array of confusion matrix values
            class_names: List of class names (defaults to indices)
            title: Plot title
            normalize: If True, normalize by row (true label)
            colorscale: Plotly colorscale name
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object with metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        cm = confusion_matrix.copy()
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Calculate metrics for binary classification
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
            # For multiclass, show overall accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            metrics_text = f"<b>Overall Accuracy:</b> {accuracy:.3f}"
        
        # Create the confusion matrix plot
        fig = ConfusionMatrixPlot.plot(
            confusion_matrix=cm,
            class_names=class_names,
            title=title,
            normalize=normalize,
            show_values=True,
            colorscale=colorscale,
            height=height,
            width=width
        )
        
        # Add metrics annotation
        fig.add_annotation(
            text=metrics_text,
            xref='paper',
            yref='paper',
            x=1.15,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
            align='left',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            borderpad=10
        )
        
        return fig


# ============================================================================
# 5. CALIBRATION PLOTS (Prediction Accuracy)
# ============================================================================

class CalibrationPlot:
    """
    Calibration plots for assessing agreement between predicted and observed outcomes.
    Works for: Survival models, logistic regression, risk prediction models, etc.
    """
    
    @staticmethod
    def plot(predicted: np.ndarray,
             observed: np.ndarray,
             title: str = 'Calibration Plot',
             xlabel: str = 'Predicted Probability',
             ylabel: str = 'Observed Probability',
             height: int = 600,
             width: int = 900,
             show_perfect: bool = True) -> go.Figure:
        """
        Create a calibration plot.
        
        Args:
            predicted: Array of predicted probabilities/values
            observed: Array of observed probabilities/values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            height: Figure height in pixels
            width: Figure width in pixels
            show_perfect: Show perfect calibration diagonal line
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Perfect calibration line
        if show_perfect:
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Perfect Calibration',
                line=dict(color='gray', dash='dash', width=2),
                hoverinfo='skip'
            ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=predicted,
            y=observed,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            height=height,
            width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig
    
    @staticmethod
    def binned_calibration(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          n_bins: int = 10,
                          strategy: str = 'quantile',
                          title: str = 'Calibration Plot',
                          height: int = 600,
                          width: int = 600) -> go.Figure:
        """
        Create a binned calibration plot (handles binning automatically).
        
        Args:
            y_true: True binary outcomes (0/1)
            y_pred: Predicted probabilities
            n_bins: Number of bins
            strategy: Binning strategy ('uniform' or 'quantile')
            title: Plot title
            height: Figure height in pixels
            width: Figure width in pixels
            
        Returns:
            Plotly Figure object
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy=strategy
        )
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2),
            hoverinfo='skip'
        ))
        
        # Model calibration
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue'),
            hovertemplate='Mean Predicted: %{x:.3f}<br>Fraction Positive: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            height=height,
            width=width,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1),
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.6, y=0.1)
        )
        
        return fig


# ============================================================================
# 6. CONVENIENCE WRAPPER
# ============================================================================

class RapidPlots:
    """
    Unified interface for all statistical plots.
    Provides convenience access to all plot types.
    """
    
    forest = ForestPlot
    residuals = ResidualPlots
    roc = ROCPlot
    confusion_matrix = ConfusionMatrixPlot
    calibration = CalibrationPlot
    
    @staticmethod
    def show_available_plots():
        """Print all available plot types."""
        print("Available Plot Types:")
        print("  - RapidPlots.forest.plot() - Forest plots for effect sizes")
        print("  - RapidPlots.residuals.residuals_vs_fitted() - Residuals vs fitted")
        print("  - RapidPlots.residuals.residuals_vs_covariate() - Residuals vs covariate")
        print("  - RapidPlots.residuals.qq_plot() - Q-Q plot for normality")
        print("  - RapidPlots.roc.plot() - Single ROC curve")
        print("  - RapidPlots.roc.compare_multiple() - Multiple ROC curves")
        print("  - RapidPlots.confusion_matrix.plot() - Confusion matrix heatmap")
        print("  - RapidPlots.confusion_matrix.plot_with_metrics() - Confusion matrix with metrics")
        print("  - RapidPlots.calibration.plot() - Calibration plot")
        print("  - RapidPlots.calibration.binned_calibration() - Binned calibration")


# ============================================================================
# 7. EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Demonstrate usage of the modular plotting utilities."""
    
    # Example 1: Forest Plot (Survival Analysis)
    hr_data = pd.DataFrame({
        'Variable': ['Age', 'Sex', 'BMI', 'Smoking', 'Diabetes'],
        'HR': [1.05, 0.85, 1.12, 1.45, 1.28],
        'CI_lower': [1.01, 0.72, 1.05, 1.22, 1.10],
        'CI_upper': [1.09, 1.01, 1.20, 1.71, 1.48]
    })
    
    fig1 = ForestPlot.plot(
        df=hr_data,
        effect_col='HR',
        lower_col='CI_lower',
        upper_col='CI_upper',
        label_col='Variable',
        title='Hazard Ratios',
        xaxis_title='Hazard Ratio'
    )
    fig1.show()
    
    # Example 2: Residual Plot (Linear Regression)
    residuals = np.random.normal(0, 1, 100)
    fitted = np.random.normal(5, 2, 100)
    
    fig2 = ResidualPlots.residuals_vs_fitted(
        residuals=residuals,
        fitted_values=fitted
    )
    fig2.show()
    
    # Example 3: ROC Curve (Classification)
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)  # Mock ROC curve
    auc = 0.75
    
    fig3 = ROCPlot.plot(fpr=fpr, tpr=tpr, auc=auc)
    fig3.show()
    
    # Example 4: Confusion Matrix
    cm = np.array([[50, 10], [5, 35]])
    class_names = ['Negative', 'Positive']
    
    fig4 = ConfusionMatrixPlot.plot_with_metrics(
        confusion_matrix=cm,
        class_names=class_names,
        title='Binary Classification Results'
    )
    fig4.show()
    
    # Example 5: Calibration Plot
    predicted = np.linspace(0, 1, 10)
    observed = predicted + np.random.normal(0, 0.05, 10)
    
    fig5 = CalibrationPlot.plot(
        predicted=predicted,
        observed=observed,
        title='Model Calibration'
    )
    fig5.show()
    
    print("All example plots created successfully!")
    return fig1, fig2, fig3, fig4, fig5


if __name__ == '__main__':
    # Show available plots
    RapidPlots.show_available_plots()
    
    # Run examples
    # example_usage()