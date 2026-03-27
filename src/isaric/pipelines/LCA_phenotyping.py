from copyreg import pickle

import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from stepmix.stepmix import StepMix

from isaric.pipelines.modules.rapid_plots import RapidPlots
from isaric.pipelines.pipeline import RAPID_BasePipeline


class RAPID_PhenotypeLCA(RAPID_BasePipeline): # Inherits from RAPID_BasePipeline if available
    """
    Pipeline for Phenotype Clustering using Latent Class Analysis (LCA).
    This class identifies hidden subgroups (phenotypes) within a population 
    based on observed categorical variables.
    """

    def __init__(self, data: pd.DataFrame, measurement_vars: list, structural_var: str = None, n_components: int = 3):
        """
        Initialize the LCA pipeline.
        
        :param data: Input DataFrame
        :param measurement_vars: List of binary/categorical variables for the measurement model
        :param structural_var: Optional variable for the structural model (e.g., outcome like HOSPITALIZ)
        :param n_components: Number of latent classes to identify
        """
        self.data = data.copy()
        self.measurement_vars = measurement_vars
        self.structural_var = structural_var
        self.n_components = n_components
        
        self.fitted_model = None
        self.clusters = None
        self.fit_metrics = None
        self.grid_results = None

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------

    def fit(self):
            """
            Public method to run the preprocessing and model fitting.
            """
            self._preprocess_data()
            self._modeling()
            self._model_evaluation()
            return self

    def summary(self) -> pd.DataFrame:
            """
            Public method to return the measurement model summary (probabilities).
            """
            if self.fitted_model is None:
                raise ValueError("Model not fitted. Call .fit() first.")
            
            self._visualization()

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------
    
    def _preprocess_data(self):
        self._data_cleaning()
        self._preprocessing()

    def _data_cleaning(self):
        """
        Logic from Notebook 2: Remove rows with missing values in key variables.
        """
        relevant_cols = self.measurement_vars + ([self.structural_var] if self.structural_var else [])
        initial_len = len(self.data)
        self.data.dropna(subset=relevant_cols, inplace=True)
        print(f"Data Cleaning: Removed {initial_len - len(self.data)} rows with NaNs.")
    
    def _preprocessing(self):
        """
        Internal: Prepares data by removing rows with missing values in key variables.
        """
        self.X = self.data[self.measurement_vars]
        self.y = self.data[self.structural_var] if self.structural_var else None


    # ------------------------------------------------------------------
    # PRIVATE METHODS: MODELING & EVALUATION
    # ------------------------------------------------------------------
    def _modeling(self):
        """
        Internal: Fits the StepMix model using Bernoulli distribution for binary variables.
        """
        self.fitted_model = StepMix(
            n_components=self.n_components, 
            measurement="bernoulli", 
            structural="bernoulli" if self.y is not None else None,
            random_state=42,
            verbose=0
        )

        self.fitted_model.fit(self.X, self.y)
        self._compute_assignments()

    def _model_evaluation(self):
        """
        Calculate Information Criteria for the current model.
        """
        if self.fitted_model:
            aic = self.fitted_model.aic(self.X, self.y)
            bic = self.fitted_model.bic(self.X, self.y)
            self.fit_metrics = pd.DataFrame({
                'Metric': ['AIC', 'BIC'],
                'Value': [aic, bic],
                'K': [self.n_components, self.n_components]
            })

    def _validation(self, cluster_range: range):
        """
        Grid search over multiple class counts.
        """
        if not hasattr(self, 'X') or self.X is None:
            self._preprocess_data()
        results = []
        for k in cluster_range:
            temp_model = StepMix(n_components=k, measurement="bernoulli", random_state=42)
            temp_model.fit(self.X, self.y)
            results.append({
                'n_clusters': k,
                'AIC': temp_model.aic(self.X, self.y),
                'BIC': temp_model.bic(self.X, self.y)
            })
        self.grid_results = pd.DataFrame(results)
        return self.grid_results

    def _visualization(self):
        """
        Orchestrate all Plotly visualizations.
        """
        self.render_profiles()
        self.render_distribution()

        if self.grid_results is not None:
            fig = RapidPlots.lca.plot_model_selection(self.grid_results)
            fig.show()   


    

    # ------------------------------------------------------------------
    # HELPER & EXPORT METHODS
    # ------------------------------------------------------------------
    def _compute_assignments(self):
        """
        Internal: Predicts classes and attaches them to the dataframe.
        """
        self.clusters = pd.Series(self.fitted_model.predict(self.X), index=self.data.index)
        self.data['latent_class'] = self.clusters

    def _render_profiles(self):
        if self.fitted_model is None: raise ValueError("Fit model first.")
        # StepMix mm_stats returns the probabilities for the measurement model
        prob_df = pd.DataFrame(self.fitted_model.get_mm_stats())
        fig = RapidPlots.lca.plot_profiles(prob_df, self.n_components)
        fig.show()

    def _render_distribution(self):
        """
        Internal method to render the class distribution bar chart.
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call .fit() first.")
            
        fig = RapidPlots.lca.plot_clusters(self.clusters)
        fig.show()
        
    def save_model(self, filepath: str):
        """
        Export model to pickle as done in Notebook 2.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.fitted_model, f)
        print(f"Model saved to {filepath}")













