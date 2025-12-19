from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


class RAPID_BaseRegression(ABC):
    """
    Abstract base class for regression analysis pipelines.
    This class provides the foundation for regression techniques as part of the ISARIC analytical pipeline,
    and generates reports useful for clinical research applied to epidemiological contexts.

    The structure is modular, allowing for future extensions into general Machine Learning pipelines.
    """
    def __init__(self, data: pd.DataFrame, outcome_str: str, predictors_list: list, regression_type: str = "Multi"):
        self._validate_inputs(data, outcome_str, predictors_list, regression_type)
        self.data = data.copy()
        self.outcome_str = outcome_str
        self.predictors_list = predictors_list
        self.regression_type = regression_type
        self._build_formula_string()
        self.preprocess_data()

    # ------------------------------------------------------------------
    # 1: PRE-PROCESSING DATA
    # ------------------------------------------------------------------
    def preprocess_data(self):
        data = self.data
        predictors_list = self.predictors_list
        #Convert categorical variables to the 'category' type
        categorical_vars = data.select_dtypes(include=['object', 'category']).columns.intersection(predictors_list)
        for var in categorical_vars:
            data[var] = data[var].astype('category')

    # ------------------------------------------------------------------
    # 2: MODEL FITTING
    # ------------------------------------------------------------------
    def fit(self, labels: dict = None):
        model = smf.glm(formula=self.formula, data=self.data, family=self.family)
        self.model_result = model.fit()
        self._setup_result_summary()

    # ------------------------------------------------------------------
    # 3: SUMMARIZATION & GRAPHICS
    # ------------------------------------------------------------------
    def summary(self):
        """Summary to be output by this regression."""
        if (self.model_result is None):
            print("Error: Model has not been fitted. Please call .fit() first.")
            return

    # ------------------------------------------------------------------
    # Property that determines statsmodel family for each regression.
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def family(self):
        """Statsmodels family used by this regression."""
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHOD (FORMULA STRING BUILDER)
    # ------------------------------------------------------------------
    def _build_formula_string(self):
        self.formula = self.outcome_str + ' ~ ' + ' + '.join(self.predictors_list)
        return
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (RESULT SUMMARY GENERATOR FOR FIT)
    # ------------------------------------------------------------------

    def _setup_result_summary(self, labels : dict = None):
        """
        Builds all generic parts of the result summary and calls
        abstract methods to build parts specific to different regression types.
        """
        result = self.model_result
        self.summary_table = result.summary2().tables[1].copy()

        self._build_result_summary_df(labels)

        self.summary_df['Study'] = self.summary_df['Study'].str.replace('T.', '')

        for col in self.summary_df.columns[1:-1]:
            self.summary_df[col] = self.summary_df[col].round(3)

        self.summary_df['p-value'] = self.summary_df['p-value'].apply(lambda x: f'{x:.4f}')
        self.summary_df = self.summary_df[self.summary_df['Study'] != 'Intercept']
        self._rename_cols_by_regression_type()

    def _map_study_label(self, df: pd.DataFrame, labels : dict = None) -> pd.DataFrame:
        if not labels:
            return df
        
        df = df.copy()
        df['Study'] = df['Study'].apply(self._parse_variable_name)
        return df

    def _parse_variable_name(self, var_name, labels : dict):
        if var_name == 'Intercept':
            return labels.get('Intercept', 'Intercept')
        elif '[' in var_name:
            base_var = var_name.split('[')[0]
            level = var_name.split('[')[1].split(']')[0]
            base_var_name = base_var.replace('C(', '').replace(')', '').strip()
            label = labels.get(base_var_name, base_var_name)
            return f'{label} ({level})'
        else:
            var_name_clean = var_name.replace('C(', '').replace(')', '').strip()
            return labels.get(var_name_clean, var_name_clean)

    @abstractmethod
    def _build_result_summary_df(self):
        """Builds the summary dataframe per linear or logistic regression."""
        pass

    @abstractmethod
    def _rename_cols_by_regression_type(self):
        """
        Renames the result summary dataframe columns from fit per regression type (linear or logistic, 
        as well as univariate or multivariate).
        """
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (PLOTS FOR SUMMARY)
    # ------------------------------------------------------------------
    def _fig_forest_plot(self,
        df, dictionary=None,
        title='Forest Plot',
        labels=['Study', 'OddsRatio', 'LowerCI', 'UpperCI'], 
        graph_id='forest-plot', graph_label='', graph_about='', only_display=False):

        # Ordering Values -> Descending Order
        df = df.sort_values(by=labels[1], ascending=True)

        # Error Handling
        if not set(labels).issubset(df.columns):
            print(df.columns)
            error_str = f'Dataframe must contain the following columns: {labels}'
            raise ValueError(error_str)

        # Prepare Data Traces
        traces = []

        # Add the point estimates as scatter plot points
        traces.append(
            go.Scatter(
                x=df[labels[1]],
                y=df[labels[0]],
                mode='markers',
                name='Odds Ratio',
                marker=dict(color='blue', size=10))
        )

        # Add the confidence intervals as lines
        for index, row in df.iterrows():
            traces.append(
                go.Scatter(
                    x=[row[labels[2]], row[labels[3]]],
                    y=[row[labels[0]], row[labels[0]]],
                    mode='lines',
                    showlegend=False,
                    line=dict(color='blue', width=2))
            )

        # Define layout
        layout = go.Layout(
            title=title,
            xaxis=dict(title='Odds Ratio'),
            yaxis=dict(
                title='', automargin=True, tickmode='array',
                tickvals=df[labels[0]].tolist(), ticktext=df[labels[0]].tolist()),
            shapes=[
                dict(
                    type='line', x0=1, y0=-0.5, x1=1, y1=len(df[labels[0]])-0.5,
                    line=dict(color='red', width=2)
                )],  # Line of no effect
            margin=dict(l=100, r=100, t=100, b=50),
            height=600
        )

        return go.Figure(data=traces, layout=layout)
    
    def _generate_forest_plot(self):
        if (self.summary_df is None and self.model_result is None):
            print("Error displaying forest plot. Please run .fit() first.")
        graph = self._fig_forest_plot(
        df = self.summary_df,
        labels = self.summary_df.columns.tolist(),
        only_display=True
        )

        return graph
    
    # ------------------------------------------------------------------
    # PRIVATE METHODS (VALIDATION)
    # ------------------------------------------------------------------

    def _validate_inputs(self, data, outcome_str, predictors_list, regression_type):
            # Validate inputs
        if data is None:
            raise ValueError("data cannot be None")
        
        if data.empty:
            raise ValueError("data cannot be empty")
        
        if outcome_str is None or outcome_str == "":
            raise ValueError("outcome_str cannot be None or empty")
        
        if predictors_list is None or len(predictors_list) == 0:
            raise ValueError("predictors_list cannot be None or empty")
        
        if regression_type is None:
            raise ValueError("regression_type cannot be None")
        
        # Check if outcome exists in data
        if outcome_str not in data.columns:
            raise ValueError(f"Outcome variable '{outcome_str}' not found in data columns")
        
        # Check if predictors exist in data
        missing_predictors = [p for p in predictors_list if p not in data.columns]
        if missing_predictors:
            raise ValueError(f"Predictor(s) not found in data columns: {missing_predictors}")
        
    # ------------------------------------------------------------------
    # PRIVATE METHODS (ASSUMPTIONS)
    # ------------------------------------------------------------------
    def _evaluate_multicolinearity(self):
        """Check whether independent variables are perfectly correlated with each other."""
        X = pd.get_dummies(self.data[self.predictors_list], drop_first=True)
        X = sm.add_constant(X)

        X = X.astype(int)

        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
        self.vif_data = vif_data[vif_data["Variable"] != "const"]

    # ------------------------------------------------------------------
    # PRIVATE METHODS (REPORT)
    # ------------------------------------------------------------------

    def _report_forest_plot(self):
        graph = self._generate_forest_plot()
        graph.show()

    def _report_multicollinearity(self, vif_threshold):
        """Report VIF and flag problematic variables"""
        print("\nVariance Inflation Factor (VIF):")
        print(self.vif_data)
        
        problematic_vif = self.vif_data[self.vif_data['VIF'] > vif_threshold]
        
        if not problematic_vif.empty:
            print(f"\nVariables with VIF > {vif_threshold}:")
            print(problematic_vif)
        else:
            print(f"\nNo variables with VIF > {vif_threshold}")