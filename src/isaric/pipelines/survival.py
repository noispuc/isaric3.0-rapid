import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score

# importing new module for plotting
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
    # ------------------------------------------------------------------
    def preprocess_data(self, df):
        """
        Handles data cleaning, one-hot encoding for categorical variables,
        and removal of missing values before model fitting.
        """
        df = df.copy()
        
        # Identify categorical columns within the predictors list
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.intersection(self.predictors)
        
        for var in categorical_vars:
            df[var] = df[var].astype('category')

        # Convert categorical variables into dummy/indicator variables
        df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
        
        # Ensure duration and event columns are numeric types
        df[self.duration_col] = pd.to_numeric(df[self.duration_col], errors='coerce')
        df[self.event_col] = pd.to_numeric(df[self.event_col], errors='coerce')

        # Update the list of predictors to include the new encoded columns
        encoded_predictors = [c for c in df.columns if c in self.predictors or any(c.startswith(p + '_') for p in categorical_vars)]
        
        # Drop rows with NaN values in required columns to prevent model failure
        all_cols = [self.duration_col, self.event_col] + encoded_predictors
        df_cox = df.dropna(subset=all_cols).copy()
        
        return df_cox[all_cols], encoded_predictors

    # ------------------------------------------------------------------
    # 2: MODEL FITTING
    # ------------------------------------------------------------------
    def fit(self, labels=None):
        """
        Trains the Cox Proportional Hazards model and prepares the Hazard Ratio table.
        """
        self.labels = labels
        self.model_data, current_predictors = self.preprocess_data(self.data)

        # Filter predictors to ensure they are valid Python identifiers for the formula
        valid_predictors = [p for p in current_predictors if str(p).isidentifier()]

        # Initialize and fit the lifelines CoxPHFitter
        self.cph_model = CoxPHFitter()
        self.cph_model.fit(
            self.model_data,
            duration_col=self.duration_col,
            event_col=self.event_col,
            formula=" + ".join(valid_predictors)
        )

        # Build the summary table with Hazard Ratios and Confidence Intervals
        summary = self.cph_model.summary.copy()
        summary['HR'] = np.exp(summary['coef'])
        summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
        summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
        
        # Format p-values for clinical publication standards
        summary['p_adj'] = summary['p'].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")

        # Reset index to turn predictors into a column
        summary_df = summary[['HR', 'CI_lower', 'CI_upper', 'p_adj']].reset_index()

        # FIX: Dynamically identify the first column (the index) and rename it to 'Variable'
        first_col_name = summary_df.columns[0] 
        summary_df.rename(columns={first_col_name: 'Variable', 'p_adj': 'p-value'}, inplace=True)

        # Apply labels if provided
        if self.labels:
            summary_df['Variable'] = summary_df['Variable'].map(self.labels).fillna(summary_df['Variable'])

        self.summary_results = summary_df

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
        # Select the first predictor for residual analysis plots
        first_var = self.model_data.columns[2]


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
