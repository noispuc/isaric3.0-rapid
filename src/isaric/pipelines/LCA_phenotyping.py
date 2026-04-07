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
        self.modelObjs = {}

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

    def summary(self):
            """
            Public method to render the measurement model summary visualizations.
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
        Required by RAPID_BasePipeline abstract class.
        Proxies to the enhanced comprehensive grid_search method.
        """
        return self.grid_search(cluster_range)

    def grid_search(self, cluster_range: range, max_iter: int = 2000):
        """
        Grid search over multiple class counts, capturing comprehensive metrics.
        """
        from sklearn.base import clone
        from isaric.pipelines.modules.rapid_assumption import ModelAssumptionTester
        
        if not hasattr(self, 'X') or self.X is None:
            self._preprocess_data()
            
        results = []
        self.modelObjs = {}
        
        base_model = StepMix(
            measurement="bernoulli", 
            structural="bernoulli" if self.y is not None else None,
            random_state=42,
            max_iter=max_iter,
            verbose=0,
            progress_bar=0
        )
        
        n_samples = self.X.shape[0]
        n_col = self.X.shape[1]
        
        for k in cluster_range:
            print(f"Testing {k} classes...")
            temp_model = clone(base_model)
            temp_model.set_params(n_components=k)
            temp_model.fit(self.X, self.y)
            
            avg_ll = temp_model.score(self.X, self.y)
            ll = avg_ll * n_samples
            npar = temp_model.n_parameters
            ncomp = temp_model.n_components
            
            aic = -2 * avg_ll * n_samples + 2 * npar
            bic = -2 * avg_ll * n_samples + npar * np.log(n_samples)
            caic = -2 * avg_ll * n_samples + npar * (np.log(n_samples) + 1)
            sabic = -2 * avg_ll * n_samples + npar * np.log(n_samples * ((n_samples + 2) / 24))
            
            entropy = temp_model.entropy(self.X)
            relentropy = 1 - entropy / (n_samples * np.log(ncomp)) if ncomp > 1 else np.nan
            
            # dof logic from AdjGridSearch5
            dof = (2**n_col) - ((ncomp - 1) + n_col * 2) 
            
            results.append({
                'n_clusters': k, 'LL': ll, 'score': avg_ll,
                'AIC': aic, 'BIC': bic, 'CAIC': caic, 'SABIC': sabic,
                'entropy': entropy, 'relative_entropy': relentropy,
                'convergence': temp_model.converged_,
                'npar': npar, 'n': n_samples, 'ncomp': ncomp, 'dof': dof
            })
            self.modelObjs[k] = temp_model
            
        stats = pd.DataFrame(results)
        
        # Likelihood ratio tests
        nested_lrt = []
        for k in range(min(cluster_range) + 1, max(cluster_range) + 1):
            if k in stats['ncomp'].values and (k-1) in stats['ncomp'].values:
                ll_k = float(stats.loc[stats['ncomp'] == k, 'LL'].values[0])
                ll_kp = float(stats.loc[stats['ncomp'] == k-1, 'LL'].values[0])
                dof_k = int(stats.loc[stats['ncomp'] == k, 'dof'].values[0])
                dof_kp = int(stats.loc[stats['ncomp'] == k-1, 'dof'].values[0])
                
                lrt = ModelAssumptionTester.likelihood_ratio_test(ll_kp, ll_k, dof_kp, dof_k)
                lrt['ncomp'] = k
                nested_lrt.append(lrt)
                
        if nested_lrt:
            stats = pd.merge(stats, pd.DataFrame(nested_lrt), on='ncomp', how='left')
            
        self.grid_results = stats
        return self.grid_results

    def decide(self, k: int):
        """
        Select definitive model to proceed.
        """
        if not hasattr(self, 'modelObjs') or k not in self.modelObjs:
            raise ValueError(f"Model with {k} clusters not found. Run grid_search first.")
        self.n_components = k
        self.fitted_model = self.modelObjs[k]
        self._compute_assignments()
        print(f"Decision stored: Model selected with k={k} clusters.")
        
    def describe(self, k: int = None):
        """
        Show exploratory plots mapping to the exploratory notebooks.
        """
        target_k = k if k is not None else self.n_components
        if hasattr(self, 'modelObjs') and target_k in self.modelObjs:
            target_model = self.modelObjs[target_k]
        else:
            if self.fitted_model is None or self.n_components != target_k:
                raise ValueError("Model not fitted.")
            target_model = self.fitted_model

        print(f"======================= LCA Description (k = {target_k}) =========================")
        df_params = target_model.get_parameters_df()
        pis_df = df_params.loc[('measurement', slice(None), 'pis')]
        prob_df = pis_df.unstack('variable')['value']
        prob_df.columns = self.measurement_vars
        
        RapidPlots.lca.plot_conditional_probs_line(prob_df).show()
        RapidPlots.lca.plot_radar_profiles(prob_df).show()
        
        target_clusters = pd.Series(target_model.predict(self.X), index=self.data.index)
        RapidPlots.lca.plot_clusters(target_clusters).show()
        
        if self.y is not None:
            print('------------------------ Cross-Tabulation (Outcome vs Predicted) -----------------')
            print(pd.crosstab(self.y, target_clusters, normalize='columns'))

    def summary_grid_plots(self):
        """
        Show the grid search criteria & entropy charts.
        """
        if self.grid_results is None:
            raise ValueError("Run grid_search first.")
        RapidPlots.lca.plot_model_selection(self.grid_results).show()
        RapidPlots.lca.plot_grid_metrics(self.grid_results).show()
        RapidPlots.lca.plot_grid_entropy(self.grid_results).show()


    def _visualization(self):
        """
        Orchestrate all Plotly visualizations.
        """
        self._render_profiles()
        self._render_distribution()

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
        #StepMix stores measurement model parameters in 'mm_stats'
        # We extract the probabilities (usually under the 'pis' key for Bernoulli)
        df_params = self.fitted_model.get_parameters_df()
        pis_df = df_params.loc[('measurement', slice(None), 'pis')]
        prob_df = pis_df.unstack('variable')['value']
        
        # Ensure the columns match your clinical variables
        prob_df.columns = self.measurement_vars
        
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













