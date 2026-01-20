import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score

# importing new module for plotting
from rapid_preprocess import RapidPreprocessor
from rapid_plots import RapidPlots


class RAPID_survival:
    """
    Pipeline that enables [Survival analysis]. 
    This class implements the technique of [survival analysis] as part of the ISARIC analytical pipeline, 
    and generates reports useful for clinical research applied to epidemiological contexts.
    
    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """

    def __init__(self, data, duration_col, event_col, predictors):
        self.data = data
        self.duration_col = duration_col
        self.event_col = event_col
        self.predictors = predictors
        
        self.cph_model = None
        self.model_data = None 
        self.summary_results = None
        self.labels = None

    # ------------------------------------------------------------------
    # 1: PRE-PROCESSING DATA
    # -----------------------------------------------------------------

    def preprocess_data(self, formula=None):
        """Generates matrices and auto-cleans problematic columns (singularities)."""
        # 1. Generate raw matrices
        y, X, _ = RapidPreprocessor.prepare_data(
            df=self.data, formula=formula,
            target_cols=[self.duration_col, self.event_col],
            predictor_cols=self.predictors, intercept=False
        )

        # 2. Drop NaNs before cleaning (prevents late-stage variance issues)
        combined = pd.concat([y, X], axis=1).dropna()
        X_clean = combined[X.columns].copy()
        y_clean = combined[y.columns].copy()

        # 3. Defensive Cleaning: Remove Constant Columns
        constant_cols = [c for c in X_clean.columns if X_clean[c].nunique() <= 1]
        if constant_cols:
            print(f"DEBUG: Dropping constant columns: {constant_cols}")
            X_clean.drop(columns=constant_cols, inplace=True)

        # 4. Defensive Cleaning: Remove Perfect Collinearity (The -1.0/1.0 issue)
        # This handles the redundant 'period' columns automatically
        if X_clean.T.duplicated().any():
            redundant = X_clean.columns[X_clean.T.duplicated()].tolist()
            print(f"DEBUG: Dropping perfectly collinear columns: {redundant}")
            X_clean = X_clean.loc[:, ~X_clean.T.duplicated()]

        self.model_data = pd.concat([y_clean, X_clean], axis=1)
        return self.model_data, X_clean.columns.tolist()
    
    # ------------------------------------------------------------------
    # 2: MODEL FITTING
    # ------------------------------------------------------------------
    def fit(self, labels=None, penalizer=0.1):
        """
        Trains the Cox Proportional Hazards model and prepares the Hazard Ratio table.
        Args:
            labels: Dictionary to map variable names to readable labels.
            penalizer: L2 regularization parameter (default 0.1 for stability).
        """
        if self.model_data is None:
            self.preprocess_data()
        
        self.labels = labels
        # Penalizer=0.1 ensures the matrix is invertible even with near-collinearity
        self.cph_model = CoxPHFitter(penalizer=penalizer)
        self.cph_model.fit(
            self.model_data, 
            duration_col=self.duration_col, 
            event_col=self.event_col
        )

        # Integrated Summary Building
        summary = self.cph_model.summary.copy()
        summary['HR'] = np.exp(summary['coef'])
        summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
        summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
        summary['p-value'] = summary['p'].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
        
        df_res = summary[['HR', 'CI_lower', 'CI_upper', 'p-value']].reset_index()
        df_res.rename(columns={df_res.columns[0]: 'Variable'}, inplace=True)
        if self.labels:
            df_res['Variable'] = df_res['Variable'].map(self.labels).fillna(df_res['Variable'])
        
        self.summary_results = df_res
    # ------------------------------------------------------------------
    # 3: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, plots=None, target_time=None):
        """
        Prints model metrics and triggers interactive visualizations.
        """
        if self.cph_model is None:
            print("Error: Model must be fitted before calling summary.")
            return

        print("\n" + "="*50)
        print("COX PROPORTIONAL HAZARDS ESTIMATES")
        print(self.summary_results.to_markdown(index=False))
        
        # Concordance Index (C-Index) represents the model's discriminative power
        print(f"\nConcordance Index (C-Index): {self.cph_model.concordance_index_:.3f}")
        print("-" * 50)
        
        if plots:
            self._generate_plots(plots, target_time)

    # ------------------------------------------------------------------
    # MÉTODOS PRIVADOS (MÉTRICAS DE FIT)
    # ------------------------------------------------------------------
    def _generate_plots(self, plots_list, target_time):
        """
        Internal dispatcher to call the RapidPlots module for specific visualizations.
        """

        if 'forest_plot' in plots_list:
            # Generate interactive Forest Plot using the new module
            RapidPlots.forest.plot(
                df=self.summary_results,
                effect_col='HR',
                lower_col='CI_lower',
                upper_col='CI_upper',
                label_col='Variable',
                title='Hazard Ratios (95% CI)'
            ).show()

        if 'roc_auc' in plots_list and target_time:
            # Generate Time-Dependent ROC curve using Plotly
            self._render_roc_plotly(target_time)
    
    def _render_roc_plotly(self, target_time):
        """
        Calculates time-dependent sensitivity and specificity for ROC visualization.
        """
        T = self.model_data[self.duration_col]
        E = self.model_data[self.event_col]
        risk_scores = self.cph_model.predict_partial_hazard(self.model_data).values

        # Define cases and controls based on the target time point
        mask = (T <= target_time) & (E == 1) | (T > target_time)
        y_true = ((T <= target_time) & (E == 1)).astype(int)
        
        # Calculate FPR and TPR for the ROC curve
        fpr, tpr, _ = roc_curve(y_true[mask], risk_scores[mask])
        auc_val = roc_auc_score(y_true[mask], risk_scores[mask])
        
        # Use the specialized ROC plotter from RapidPlots
        RapidPlots.roc.plot(fpr, tpr, auc_val, title=f'ROC Curve at t={target_time}').show()