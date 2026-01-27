import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from sklearn.metrics import roc_curve, roc_auc_score

# importing new module for plotting
from rapid_preprocess import RapidPreprocessor
from rapid_plots import RapidPlots
from pipeline import RAPID_Pipeline


class RAPID_survival(RAPID_Pipeline):
    """
    Pipeline that enables [Survival analysis]. 
    This class implements the technique of [survival analysis] as part of the ISARIC analytical pipeline, 
    and generates reports useful for clinical research applied to epidemiological contexts.
    
    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    This class inherits from RAPID_Pipeline and implements all required abstract methods.
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
        """
        Triggers the internal cleaning and preprocessing workflow.
        """
        self._data_cleaning()
        self._preprocessing(formula)
        return self.model_data
    
    # ------------------------------------------------------------------
    # 2: MODEL FITTING
    # ------------------------------------------------------------------
    def fit(self, labels=None, penalizer=0.1):
        """
        Executes the modeling and evaluation sequence.
        """
        if self.model_data is None:
            self.preprocess_data()
        
        self.labels = labels
        self._modeling(penalizer)
        self._model_evaluation()
    # ------------------------------------------------------------------
    # 3: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self, plots=None, target_time=None):
        """
        Reports model findings and generates visualizations.
        """
        if self.cph_model is None:
            print("Error: Model must be fitted before calling summary.")
            return

        print("\n" + "="*50)
        print("COX PROPORTIONAL HAZARDS ESTIMATES")
        print(self.summary_results.to_markdown(index=False))
        print(f"\nConcordance Index (C-Index): {self.cph_model.concordance_index_:.3f}")
        
        if plots:
            self._visualization(plots, target_time)

    # ------------------------------------------------------------------
    # PRIVATE METHODS (REQUIRED BY RAPID_Pipeline)
    # ------------------------------------------------------------------
    def _data_cleaning(self):
        """
        Handles initial data sanitization and missing values.
        """
        required_cols = [self.duration_col, self.event_col] + self.predictors
        self.data = self.data.dropna(subset=required_cols)
    
    def _preprocessing(self, formula):
        """
        Converts raw data into design matrices and handles collinearity.
        """
        y, X, _ = RapidPreprocessor.prepare_data(
            df=self.data, formula=formula,
            target_cols=[self.duration_col, self.event_col],
            predictor_cols=self.predictors, intercept=False
        )

        # Drop zero-variance columns
        X = X.loc[:, X.nunique() > 1]
        # Drop perfectly collinear columns
        X = X.loc[:, ~X.T.duplicated()]

        self.model_data = pd.concat([y, X], axis=1)

    def _modeling(self, penalizer):
        """
        Fits the survival model using the Cox Proportional Hazards algorithm.
        """
        self.cph_model = CoxPHFitter(penalizer=penalizer)
        self.cph_model.fit(
            self.model_data, 
            duration_col=self.duration_col, 
            event_col=self.event_col
        )

    def _model_evaluation(self):
        """
        Generates statistical summaries and hazard ratios.
        """
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

    def _validation(self):
        """
        Performs model validation (to be implemented).
        """
        pass

    def _visualization(self, plots_list, target_time):
        """
        Manages the generation of forest plots and ROC curves.
        """
        if 'forest_plot' in plots_list:
            # The error happened here. We must pass the column names explicitly.
            RapidPlots.forest.plot(
                df=self.summary_results,
                effect_col='HR',          # The Hazard Ratio column
                lower_col='CI_lower',     # The Confidence Interval Lower Bound
                upper_col='CI_upper',     # The Confidence Interval Upper Bound
                label_col='Variable',     # The Variable names
                title='Hazard Ratios (95% CI)'
            ).show()

        if 'roc_auc' in plots_list and target_time:
            self._render_roc_plotly(target_time)

    def _render_roc_plotly(self, target_time):
        """
        Internal helper for ROC calculation and rendering.
        """
        T = self.model_data[self.duration_col]
        E = self.model_data[self.event_col]
        risk_scores = self.cph_model.predict_partial_hazard(self.model_data).values
        
        mask = (T <= target_time) & (E == 1) | (T > target_time)
        y_true = ((T <= target_time) & (E == 1)).astype(int)
        
        fpr, tpr, _ = roc_curve(y_true[mask], risk_scores[mask])
        auc_val = roc_auc_score(y_true[mask], risk_scores[mask])
        
        RapidPlots.roc.plot(fpr, tpr, auc_val, title=f'ROC at t={target_time}').show()