import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objs as go
import warnings
import math


from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ApproximationWarning
from sklearn.metrics import roc_curve, roc_auc_score

# --- Classe Principal: RAPID_survival (Consolidada e Modularizada) ---

class RAPID_survival:
    """
    Pipeline that enables [Survival analysis]. 
    This class implements the technique of [survival analysis] as part of the ISARIC analytical pipeline, 
    and generates reports useful for clinical research applied to epidemiological contexts.
    
    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """

    def __init__(self, data: pd.DataFrame, duration_col: str, event_col: str, predictors: list):
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predictors = predictors
        self.method = "CoxPH"

        self.cph_model = None
        self.model_data = None # DataFrame pre-processado e limpo (usado no fit)
        self.summary_results = None
        self.labels = None

    # ------------------------------------------------------------------
    # FASE 1: PRE-PROCESSAMENTO
    # ------------------------------------------------------------------
    def preprocess_data(self, df):
        """Internal method to handle data cleaning and encoding."""
        df = df.copy()

        # 1. Handle categorical variables (One-Hot Encoding)
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.intersection(self.predictors)
        for var in categorical_vars:
            df[var] = df[var].astype('category')

        df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

        # 2. Ensure numerical types
        df[self.duration_col] = pd.to_numeric(df[self.duration_col], errors='coerce')
        df[self.event_col] = pd.to_numeric(df[self.event_col], errors='coerce')

        # 3. Update predictors to include one-hot encoded columns
        encoded_predictors = [c for c in df.columns if c in self.predictors or any(c.startswith(p + '_') for p in categorical_vars)]
        
        # 4. Remove rows with missing values
        all_cols_to_check = [self.duration_col, self.event_col] + encoded_predictors
        df_cox = df.dropna(subset=all_cols_to_check).copy()
        
        df_cox = df_cox[[self.duration_col, self.event_col] + encoded_predictors]
        
        return df_cox, encoded_predictors

    # ------------------------------------------------------------------
    # FASE 2: FIT DO MODELO
    # ------------------------------------------------------------------
    def fit(self, labels: dict = None):
        """
        Fits the model, (in this casse the Cox Proportional Hazards model) using the pre-specified data and predictors.
        Stores the fitted model and summary results internally.
        """
        self.labels = labels
        self.model_data, current_predictors = self.preprocess_data(self.data)

        # Validate predictors to ensure they are valid for the formula
        valid_predictors = [p for p in current_predictors if isinstance(p, str) and p.isidentifier()]

        if not valid_predictors:
            raise ValueError("No valid predictors found for the formula.")

        self.cph_model = CoxPHFitter()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.cph_model.fit(
                self.model_data,
                duration_col=self.duration_col,
                event_col=self.event_col,
                formula=" + ".join(valid_predictors)
            )

        summary = self.cph_model.summary.copy()
        summary['HR'] = np.exp(summary['coef'])
        summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
        summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
        
        summary['p_adj'] = summary['p'].apply(lambda p: "<0.001" if p < 0.001 else round(p, 3))

        summary_df = summary[['HR', 'CI_lower', 'CI_upper', 'p_adj']].reset_index()
        summary_df.rename(columns={'index': 'Variable', 'p_adj': 'p-value'}, inplace=True)

        if self.labels:
            summary_df['Variable'] = summary_df['Variable'].map(self.labels).fillna(summary_df['Variable'])

        self.summary_results = summary_df
        
        print(f"Cox PH model fitted successfully on {len(self.model_data)} observations.")

    # ------------------------------------------------------------------
    # FASE 3: SUMMARIZATION & DIAGNÓSTICO
    # ------------------------------------------------------------------
    def summary(self, fit_measures: bool = True, plots: list = None, target_time: float = None):
        """
        Reports the results of the survival analysis, generating publication-ready tables and plots.

        Args:
            fit_measures (bool): If fit measures (AIC, BIC, C-index) will be reported.
            plots (list, optional): List of plots to be displayed. Options: 'forest_plot', 
                                    'schoenfeld_residuals', 'martingale_residuals',
                                    'deviance_residuals', 'roc_auc', 'calibration_plot'.
            target_time (float, optional): The time point for time-dependent metrics (ROC, Calibration).
        """
        if self.cph_model is None:
            print("Error: Model has not been fitted. Please call .fit() first.")
            return

        print("## Cox Proportional Hazards Model Summary")
        print("-" * 50)
        
        # 1. Parameter Table (Hazard Ratios)
        print("### Hazard Ratio Estimates (Publication-Ready Table)")
        print(self.summary_results.to_markdown(index=False, floatfmt=".3f"))
        print("-" * 50)

        # 2. Fit Measures
        if fit_measures:
            print("### Model Fit and Discrimination Metrics")
            self._display_model_fit_metrics()
            self._calculate_c_index()
            print("-" * 50)

        # 3. Plots
        if plots:
            self._generate_plots(plots, target_time)
            print("-" * 50)

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (MÉTRICAS DE FIT)
    # ------------------------------------------------------------------
    def _display_model_fit_metrics(self):
        """Calculates and displays the partial AIC and BIC for a Cox model."""
        try:
            print(f"Partial AIC: {self.cph_model.AIC_partial_:.2f}")

            n = self.model_data.shape[0]
            k = len(self.cph_model.params_)
            log_partial_likelihood = self.cph_model.log_likelihood_

            if not math.isnan(log_partial_likelihood):
                bic_partial = -2 * log_partial_likelihood + k * np.log(n)
                print(f"Partial BIC: {bic_partial:.2f}")
            else:
                print("Partial BIC: N/A (Log-likelihood is undefined)")

        except Exception as e:
            print(f"Error during fit metrics calculation: {e}")

    def _calculate_c_index(self):
        """Calculates and prints the Concordance Index (C-Index) of the model."""
        

        c_index = concordance_index(
            event_times=self.model_data[self.duration_col],
            predicted_scores=self.cph_model.predict_partial_hazard(self.model_data),
            event_observed=self.model_data[self.event_col]
        )
        print(f"Concordance Index (C-Index): {c_index:.3f}")

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (LÓGICA DE PLOTAGEM GERAL)
    # ------------------------------------------------------------------
    def _get_original_predictor(self, encoded_predictor):
        """Helper to try and map an encoded predictor name back to its original name in self.predictors."""
        for original in self.predictors:
            if original == encoded_predictor:
                return original # É uma variável binária/contínua não-encoded
            if encoded_predictor.startswith(original + '_'):
                return original
        return encoded_predictor # Retorna o nome codificado se não encontrar o original

    def _generate_plots(self, plots_list: list, target_time: float):
        """Handles the internal logic for plot generation, mapping plot names to methods."""


        # Safety check: ensure summary_results has 'Variable' column
        if 'Variable' not in self.summary_results.columns:
            print("Error: 'Variable' column not found in summary_results.")
            print("Available columns:", self.summary_results.columns.tolist())
            return
    

        # Tenta usar a primeira variável do summary (que pode ser encoded ou original)
        first_summary_var = self.summary_results['Variable'].iloc[0]
        # Mapeia para o nome da coluna original (se possível) para plots de resíduos univariados
        plot_covariate = self._get_original_predictor(first_summary_var)



        print("### Graphical Diagnostics and Validation")

        if 'forest_plot' in plots_list:
            print("Generating Forest Plot...")
            self._plot_forest_plot()

        if 'schoenfeld_residuals' in plots_list:
            print("Generating Schoenfeld Residuals...")
            self._plot_schoenfeld_residuals(plot_covariate)

        if 'martingale_residuals' in plots_list:
            print("Generating Martingale Residuals...")
            self._plot_martingale_residuals(plot_covariate)

        if 'deviance_residuals' in plots_list:
            print("Generating Deviance Residuals...")
            self._plot_deviance_residuals(plot_covariate)

        if 'roc_auc' in plots_list and target_time is not None:
            print(f"Generating ROC/AUC at t={target_time}...")
            self._plot_roc_auc_manual(target_time)

        if 'calibration_plot' in plots_list and target_time is not None:
            print(f"Generating Calibration Plot at t={target_time}...")
            self._plot_calibration(target_time)

        if target_time is None and ('roc_auc' in plots_list or 'calibration_plot' in plots_list):
            print("Warning: target_time é necessário para ROC/AUC e Calibration Plot, mas não foi fornecido.")


    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (PLOTS ESPECÍFICOS)
    # ------------------------------------------------------------------

    def _plot_forest_plot(self):
        """Generates and displays the Forest Plot of Hazard Ratios."""
        df = self.summary_results.copy()
        labels=['Variable', 'HR', 'CI_lower', 'CI_upper']
        title='Forest Plot of Hazard Ratios'

        df = df.sort_values(by=labels[1], ascending=True).copy()

        traces = []

        traces.append(
            go.Scatter(
                x=df[labels[1]],
                y=df[labels[0]],
                mode='markers',
                name='Hazard Ratio',
                marker=dict(color='blue', size=10))
        )

        for index, row in df.iterrows():
            traces.append(
                go.Scatter(
                    x=[row[labels[2]], row[labels[3]]],
                    y=[row[labels[0]], row[labels[0]]],
                    mode='lines',
                    showlegend=False,
                    line=dict(color='blue', width=2))
            )

        layout = go.Layout(
            title=title,
            xaxis=dict(title='Hazard Ratio'),
            yaxis=dict(
                title='', automargin=True, tickmode='array',
                tickvals=df[labels[0]].tolist(), ticktext=df[labels[0]].tolist()),
            shapes=[
                dict(
                    type='line', x0=1, y0=-0.5, x1=1, y1=len(df[labels[0]])-0.5,
                    line=dict(color='red', width=2)
                )],
            margin=dict(l=200, r=100, t=100, b=50),
            height=600
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.show()

    def _plot_schoenfeld_residuals(self, covariate_name):
        """Runs the PH test and plots Schoenfeld residuals for a specific covariate/model."""
        try:
            print("--- Proportional Hazards Assumption Test Results (All Covariates) ---")
            
            # 1. Teste de PH
            self.cph_model.check_assumptions(self.model_data.copy(), show_plots=False, p_value_threshold=0.05)
            
            print(f"\n--- Plotting Schoenfeld Residuals (PH Check) ---")
            
            # Tenta encontrar a versão codificada da variável para plotagem
            covariate_to_plot_encoded = next(
                (col for col in self.cph_model.covariate_names_ if col.startswith(covariate_name + '_')),
                covariate_name
            )
            
            if covariate_to_plot_encoded in self.cph_model.params_.index:
                # Usa o método nativo do lifelines para plotar os residuais de Schoenfeld
                self.cph_model.plot_residuals_vs_time(covariate=covariate_to_plot_encoded, 
                                                      kind='schoenfeld', 
                                                      show_plots=True)
            else:
                # Se não for uma variável univariada, plota o resultado global ou avisa
                print(f"Variável '{covariate_name}' não é um preditor direto. Plotando a primeira variável para referência.")
                first_encoded_var = self.cph_model.covariate_names_[0] if self.cph_model.covariate_names_ else None
                if first_encoded_var:
                     self.cph_model.plot_residuals_vs_time(covariate=first_encoded_var, 
                                                      kind='schoenfeld', 
                                                      show_plots=True)


        except Exception as e:
            print(f"An error occurred during Schoenfeld residual analysis: {e}")

    def _plot_martingale_residuals(self, covariate_name):
        """Calculates and plots Martingale residuals against a covariate."""
        try:
            # Usa o método robusto compute_residuals do lifelines
            martingale_residuals = self.cph_model.compute_residuals(self.model_data, kind='martingale')
            
            plot_df = pd.DataFrame({
                'martingale_residuals': martingale_residuals.squeeze(),
                covariate_name: self.model_data[covariate_name]
            })

            plt.figure(figsize=(10, 6))
            # Usa boxplot para categóricas, scatterplot para contínuas
            if plot_df[covariate_name].dtype.name in ['category', 'object', 'bool'] or len(plot_df[covariate_name].unique()) < 10:
                sns.boxplot(x=covariate_name, y='martingale_residuals', data=plot_df)
            else:
                sns.scatterplot(x=covariate_name, y='martingale_residuals', data=plot_df, alpha=0.6)
                lowess = sm.nonparametric.lowess(plot_df['martingale_residuals'], plot_df[covariate_name], frac=0.3)
                plt.plot(lowess[:, 0], lowess[:, 1], color='orange', linestyle='-', linewidth=3, label='LOWESS Smoother')
            
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Martingale Residuals by {covariate_name}')
            plt.xlabel(covariate_name)
            plt.ylabel('Martingale Residuals')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()

        except Exception as e:
            print(f"An error occurred while plotting Martingale residuals: {e}")

    def _plot_deviance_residuals(self, covariate_name):
        """Calculates and plots Deviance residuals against a covariate."""
        try:
            # Usa o método robusto compute_residuals do lifelines
            deviance_residuals = self.cph_model.compute_residuals(self.model_data, kind='deviance')
            
            plot_df = pd.DataFrame({
                'deviance_residuals': deviance_residuals.squeeze(),
                covariate_name: self.model_data[covariate_name]
            }).dropna(subset=['deviance_residuals'])

            plt.figure(figsize=(10, 6))
            if plot_df[covariate_name].dtype.name in ['category', 'object', 'bool'] or len(plot_df[covariate_name].unique()) < 10:
                sns.boxplot(x=covariate_name, y='deviance_residuals', data=plot_df)
            else:
                 sns.scatterplot(x=covariate_name, y='deviance_residuals', data=plot_df, alpha=0.6)
                 lowess = sm.nonparametric.lowess(plot_df['deviance_residuals'], plot_df[covariate_name], frac=0.3)
                 plt.plot(lowess[:, 0], lowess[:, 1], color='orange', linestyle='-', linewidth=3, label='LOWESS Smoother')
                 
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Deviance Residuals by {covariate_name}')
            plt.xlabel(covariate_name)
            plt.ylabel('Deviance Residuals')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()

        except Exception as e:
            print(f"An error occurred while plotting Deviance residuals: {e}")

    def _plot_roc_auc_manual(self, target_time, plot_only=True):
        """Calculates and plots the Time-Dependent ROC curve and AUC (from notebook)."""
        T = self.model_data[self.duration_col]
        E = self.model_data[self.event_col]
        risk_scores = self.cph_model.predict_partial_hazard(self.model_data).values

        cases_mask = (T <= target_time) & (E == 1)
        controls_mask = T > target_time
        relevant_subjects_mask = cases_mask | controls_mask

        if not relevant_subjects_mask.any():
            print(f"Error: No subjects available for comparison at t={target_time}.")
            return

        y_true = cases_mask[relevant_subjects_mask].astype(int)
        y_score = risk_scores[relevant_subjects_mask]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_value:.3f})')
        plt.fill_between(fpr, tpr, alpha=0.2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.5)')

        plt.title(f'Time-Dependent ROC Curve at t={target_time}')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.axis('square')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    def _plot_calibration(self, target_time):
        """Calculates and plots a Calibration Plot (from notebook)."""
        predicted_survival = self.cph_model.predict_survival_function(self.model_data, times=[target_time]).squeeze()

        calib_df = pd.DataFrame({
            'predicted_survival': predicted_survival,
            'duration': self.model_data[self.duration_col],
            'event': self.model_data[self.event_col]
        })

        try:
            calib_df['decile'] = pd.qcut(calib_df['predicted_survival'], 10, labels=False, duplicates='drop')
        except ValueError:
            try:
                calib_df['decile'] = pd.qcut(calib_df['predicted_survival'], 5, labels=False, duplicates='drop')
            except ValueError:
                print("Error: Too few unique survival probabilities to create meaningful bins for calibration.")
                return

        observed_probs = []
        predicted_probs = []

        for i in sorted(calib_df['decile'].unique()):
            decile_df = calib_df[calib_df['decile'] == i]
            mean_predicted = decile_df['predicted_survival'].mean()

            kmf = KaplanMeierFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ApproximationWarning)
                kmf.fit(decile_df['duration'], event_observed=decile_df['event'])

            observed_survival = kmf.predict(target_time, interpolate=True)

            if isinstance(observed_survival, pd.Series):
                observed_survival = observed_survival.iloc[0]

            predicted_probs.append(mean_predicted)
            observed_probs.append(observed_survival)

        # Plot the results
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(predicted_probs, observed_probs, 'o-', ms=6, label='Model Calibration')

        ax.set_xlabel(f'Predicted Survival Probability at t={target_time}')
        ax.set_ylabel('Observed Survival Fraction (Kaplan-Meier)')
        ax.set_title(f'Calibration Plot at Time t={target_time}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.show()