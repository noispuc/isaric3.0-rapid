import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as    sns
import statsmodels.api as sm
import plotly.graph_objs as go
import warnings
import math

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ApproximationWarning
from sklearn.metrics import roc_curve, roc_auc_score




class RAPID_survival:
    """
    Pipeline that enables [Survival analysis]. 
    This class implements the technique of [survival analysis] as part of the ISARIC analytical pipeline, and generate reports useful for clinical research applied to epidemiological contexts.
    Attributes:
        data (pd.DataFrame): The input dataset in pandas DataFrame format.
        duration_col (str): The column name for time-to-event/follow-up.
        event_col (str): The column name for the event indicator (0=censored, 1=event).
        predictors (list): List of predictor column names.
        method (str): The survival analysis method used ("CoxPH").
        cph_model (CoxPHFitter): The fitted Cox Proportional Hazards model object.
        model_data (pd.DataFrame): The preprocessed data used for model fitting.
        summary_results (pd.DataFrame): DataFrame containing HR, CI, and p-values.
    """

    def __init__(self, data: pd.DataFrame, duration_col: str, event_col: str, predictors: list):
        """
        Initializes the survival analysis pipeline with access to the dataset and 
        the primary columns (duration, event, and predictors).
        
        Args:
            data (pd.DataFrame): [The dataset in pandas DataFrame format].
            duration_col (str): [The column name for time-to-event or follow-up time].
            event_col (str): [The column name for the binary event indicator (0 or 1)].
            predictors (list): [A list of column names used as predictors (covariates)].
        """
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predictors = predictors
        self.method = "CoxPH"

        # Will be set after calling .fit()
        self.cph_model = None
        self.model_data = None
        self.summary_results = None
        self.labels = None

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

    def fit(self, labels: dict = None):
        """
        Fits the Cox Proportional Hazards model using the pre-specified data and predictors.

        Args:
            labels (dict, optional): [A dictionary to map predictor names to
                                     human-readable labels for output]. Defaults to None.

        """
        self.labels = labels
        # call the correct preprocessing method
        self.model_data, current_predictors = self.preprocess_data(self.data)

        # 5. Fit the Cox model
        self.cph_model = CoxPHFitter()
        with warnings.catch_warnings():
            # Suppress RuntimeWarning that may occur with extreme HRs
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            self.cph_model.fit(
                self.model_data,
                duration_col=self.duration_col,
                event_col=self.event_col,
                formula=" + ".join(current_predictors)
            )

        # 6. Calculate summary results
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

    def summary(self, fit_measures: bool = True, plots: list = None, target_time: float = None):
        """
        Reports the results of the survival analysis conducted in the pipeline, 
        generating publication-ready tables and plots, using the ISARIC standards.

        Args:
            fit_measures (bool): [if fit measures (AIC, BIC, C-index) will be reported, default=True].
            plots (list, optional): [plots to be displayed]. Alternatives include 'forest_plot', 'schoenfeld_residuals', 'martingale_residuals','deviance_residuals', 'roc_auc', 'calibration_plot'.
            target_time (float, optional): [The specific time point for time-dependent ,metrics (ROC, Calibration)]. Required for time-dependent plots/metrics.
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
            if target_time:
                self._plot_roc_auc_manual(target_time=target_time, plot_only=False)
            print("-" * 50)

        # 3. Plots
        if plots:
            self._generate_plots(plots, target_time)
            print("-" * 50)

    def _generate_plots(self, plots_list: list, target_time: float):
        """Handles the internal logic for plot generation."""
        diagnostic = CoxPHDiagnostic(self.cph_model, self.model_data, self.duration_col, self.event_col)

        print("### Graphical Diagnostics and Validation")

        if 'forest_plot' in plots_list:
            print("Generating Forest Plot...")
            CoxPHAnalyzer.fig_forest_plot(
                self.summary_results,
                title="Forest Plot of Hazard Ratios",
                only_display=True
            )

        if 'schoenfeld_residuals' in plots_list and self.predictors:
            # For simplicity, plot the first predictor if Schoenfeld is requested
            # In a real pipeline, the user might specify which one to plot,
            # or we might run the full PH test and plot violators.
            first_predictor = self.summary_results['Variable'].iloc[0]
            if first_predictor in self.model_data.columns:
                diagnostic.test_and_plot_schoenfeld(first_predictor)
            else:
                print(f"Cannot plot Schoenfeld for '{first_predictor}': it might be an encoded variable. Please inspect the summary table.")


        if 'martingale_residuals' in plots_list and self.predictors:
            # For simplicity, plot the first predictor if Martingale is requested
            first_predictor = self.summary_results['Variable'].iloc[0]
            if first_predictor in self.model_data.columns:
                diagnostic.plot_martingale_residuals_manual(first_predictor)
            else:
                print(f"Cannot plot Martingale for '{first_predictor}': it might be an encoded variable. Please inspect the summary table.")

        if 'deviance_residuals' in plots_list and self.predictors:
            # For simplicity, plot the first predictor if Deviance is requested
            first_predictor = self.summary_results['Variable'].iloc[0]
            if first_predictor in self.model_data.columns:
                diagnostic.plot_deviance_residuals_manual(first_predictor)
            else:
                print(f"Cannot plot Deviance for '{first_predictor}': it might be an encoded variable. Please inspect the summary table.")

        if 'roc_auc' in plots_list and target_time:
            diagnostic.plot_roc_auc_manual(target_time)

        if 'calibration_plot' in plots_list and target_time:
            diagnostic.plot_calibration(target_time)


    # --- Methods from CoxPHDiagnostic integrated as private methods ---

    def _display_model_fit_metrics(self):
        """Calculates and displays the partial AIC and BIC for a Cox model."""
        try:
            print(f"Partial AIC: {self.cph_model.AIC_partial_:.2f}")

            # Partial BIC = -2 * log-likelihood + k * log(n)
            n = self.model_data.shape[0]
            k = len(self.cph_model.params_)
            log_partial_likelihood = self.cph_model.log_likelihood_

            # Check if log_partial_likelihood is not math.nan (can happen with numerical issues)
            if not math.isnan(log_partial_likelihood):
                bic_partial = -2 * log_partial_likelihood + k * np.log(n)
                print(f"Partial BIC: {bic_partial:.2f}")
            else:
                print("Partial BIC: N/A (Log-likelihood is undefined)")

        except Exception as e:
            print(f"Error during fit metrics calculation: {e}")

    def _calculate_c_index(self):
        """Calculates and prints the Concordance Index (C-Index) of the model."""
        predicted_scores = self.cph_model.predict_partial_hazard(self.model_data)

        c_index = concordance_index(
            durations=self.model_data[self.duration_col],
            scores=predicted_scores,
            event_observed=self.model_data[self.event_col]
        )

        print(f"Concordance Index (C-Index): {c_index:.3f}")

    def _plot_roc_auc_manual(self, target_time, plot_only=True):
        """Calculates and plots the Time-Dependent ROC curve and AUC (internal helper)."""
        diagnostic = CoxPHDiagnostic(self.cph_model, self.model_data, self.duration_col, self.event_col)
        diagnostic.plot_roc_auc_manual(target_time)

    def _plot_calibration(self, target_time):
        """Calculates and plots the Calibration Plot (internal helper)."""
        diagnostic = CoxPHDiagnostic(self.cph_model, self.model_data, self.duration_col, self.event_col)
        diagnostic.plot_calibration(target_time)

    # --- Auxiliary Diagnostic Class (Same as before, used internally by _generate_plots) ---

class CoxPHDiagnostic:
    """
    Auxiliary class containing all external diagnostic methods for Cox PH.
    This class is instantiated and used internally by RAPID_survival.
    """

    def __init__(self, cph_model, dataframe, duration_col, event_col):
        self.cph_model = cph_model
        self.dataframe = dataframe
        self.duration_col = duration_col
        self.event_col = event_col

    def test_and_plot_schoenfeld(self, covariate_to_plot):
        """Runs the PH test and plots Schoenfeld residuals for a specific covariate."""
        try:
            print("--- Proportional Hazards Assumption Test Results (All Covariates) ---")

            self.cph_model.check_assumptions(self.dataframe, show_plots=False, p_value_threshold=0.05)

            print(f"\n--- Plotting Schoenfeld Residuals for: '{covariate_to_plot}' ---")

            if covariate_to_plot in self.cph_model.params_.index:
                residuals_df = self.cph_model.compute_residuals(self.dataframe, kind='schoenfeld')

                relevant_residuals = residuals_df.columns[residuals_df.columns.str.startswith(covariate_to_plot)]

                if not relevant_residuals.empty:
                    res_col = relevant_residuals[0]
                    residuals = residuals_df[res_col]
                    times = residuals_df.index.values

                    lowess = sm.nonparametric.lowess(residuals, times, frac=0.3)

                    plt.figure(figsize=(10, 6))
                    plt.scatter(times, residuals, marker='.', alpha=0.5, label='Schoenfeld Residuals')
                    plt.plot(lowess[:, 0], lowess[:, 1], color='orange', linestyle='-', linewidth=3, label='LOWESS Smoother')

                    plt.axhline(y=0, color='k', linestyle='--')
                    plt.title(f'Schoenfeld Residuals for {res_col}')
                    plt.xlabel('Time')
                    plt.ylabel(f'Residuals for {res_col}')
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.legend()
                    plt.show()
                else:
                    print(f"Schoenfeld residuals not found for variable: '{covariate_to_plot}'.")
            else:
                print(f"Variable '{covariate_to_plot}' not found in the fitted model parameters.")

        except Exception as e:
            print(f"An error occurred during Schoenfeld residual analysis: {e}")

    def plot_martingale_residuals_manual(self, covariate_name):
        """Calculates and plots Martingale residuals against a categorical covariate."""
        try:
            E = self.dataframe[self.event_col].values
            risk_scores = self.cph_model.predict_partial_hazard(self.dataframe).values
            baseline_hazard_df = self.cph_model.baseline_cumulative_hazard_

            baseline_hazard_times = baseline_hazard_df.index.values
            baseline_hazard_values = baseline_hazard_df.iloc[:, 0].values

            indices = np.searchsorted(baseline_hazard_times, self.dataframe[self.duration_col].values, side='right') - 1
            indices = np.maximum(0, indices)

            cumulative_hazard_at_T = baseline_hazard_values[indices]
            cumulative_subject_hazard = cumulative_hazard_at_T * risk_scores

            martingale_residuals = E - cumulative_subject_hazard

            plot_df = pd.DataFrame({
                'martingale_residuals': martingale_residuals,
                covariate_name: self.dataframe[covariate_name]
            })

            plt.figure(figsize=(8, 6))
            sns.boxplot(x=covariate_name, y='martingale_residuals', data=plot_df)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Martingale Residuals by {covariate_name}')
            plt.xlabel(covariate_name)
            plt.ylabel('Martingale Residuals')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()

        except Exception as e:
            print(f"An error occurred while plotting Martingale residuals: {e}")

    def plot_deviance_residuals_manual(self, covariate_name):
        """Calculates and plots Deviance residuals against a categorical covariate."""
        try:
            E = self.dataframe[self.event_col].values
            risk_scores = self.cph_model.predict_partial_hazard(self.dataframe).values
            baseline_hazard_df = self.cph_model.baseline_cumulative_hazard_

            baseline_hazard_times = baseline_hazard_df.index.values
            baseline_hazard_values = baseline_hazard_df.iloc[:, 0].values

            indices = np.searchsorted(baseline_hazard_times, self.dataframe[self.duration_col].values, side='right') - 1
            indices = np.maximum(0, indices)

            cumulative_hazard_at_T = baseline_hazard_values[indices]
            cumulative_subject_hazard = cumulative_hazard_at_T * risk_scores

            martingale_residuals = E - cumulative_subject_hazard

            deviance_residuals = np.zeros_like(martingale_residuals, dtype=float)

            for i, (r_m, d_i) in enumerate(zip(martingale_residuals, E)):
                if d_i == 0:
                    deviance_residuals[i] = -np.sign(r_m) * np.sqrt(2 * -r_m)
                else:
                    if d_i - r_m <= 0:
                        deviance_residuals[i] = np.nan
                    else:
                        term = d_i * np.log(d_i / (d_i - r_m)) - r_m
                        deviance_residuals[i] = np.sign(r_m) * np.sqrt(2 * term)

            plot_df = pd.DataFrame({
                'deviance_residuals': deviance_residuals,
                covariate_name: self.dataframe[covariate_name]
            }).dropna(subset=['deviance_residuals'])

            plt.figure(figsize=(8, 6))
            sns.boxplot(x=covariate_name, y='deviance_residuals', data=plot_df)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'Deviance Residuals by {covariate_name}')
            plt.xlabel(covariate_name)
            plt.ylabel('Deviance Residuals')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.show()

        except Exception as e:
            print(f"An error occurred while plotting Deviance residuals: {e}")

    def plot_roc_auc_manual(self, target_time):
        """Calculates and plots the Time-Dependent ROC curve and AUC."""
        try:
            T = self.dataframe[self.duration_col]
            E = self.dataframe[self.event_col]
            risk_scores = self.cph_model.predict_partial_hazard(self.dataframe).values

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

        except Exception as e:
            print(f"An error occurred during ROC/AUC plotting: {e}")

    def plot_calibration(self, target_time):
        """Calculates and plots a Calibration Plot."""
        try:
            predicted_survival = self.cph_model.predict_survival_function(self.dataframe, times=[target_time]).squeeze()

            calib_df = pd.DataFrame({
                'predicted_survival': predicted_survival,
                'duration': self.dataframe[self.duration_col],
                'event': self.dataframe[self.event_col]
            })

            try:
                calib_df['decile'] = pd.qcut(calib_df['predicted_survival'], 10, labels=False, duplicates='drop')
            except ValueError:
                print("Warning: Could not create 10 deciles. Falling back to 5 bins.")
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
        except Exception as e:
            print(f"An error occurred during calibration plotting: {e}")

    def display_model_fit_metrics(self):
        """Calculates and displays the partial AIC and BIC."""
        try:
            print(f"Partial AIC: {self.cph_model.AIC_partial_:.2f}")

            n = self.dataframe.shape[0]
            k = len(self.cph_model.params_)
            log_partial_likelihood = self.cph_model.log_likelihood_

            if not math.isnan(log_partial_likelihood):
                bic_partial = -2 * log_partial_likelihood + k * np.log(n)
                print(f"Partial BIC: {bic_partial:.2f}")
            else:
                print("Partial BIC: N/A (Log-likelihood is undefined)")

        except Exception as e:
            print(f"An error occurred during fit metrics calculation: {e}")