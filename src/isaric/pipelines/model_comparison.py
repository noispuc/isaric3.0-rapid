import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
from isaric.pipelines.regression import RAPID_BaseRegression

class RAPID_ModelComparator:
    def __init__(self, models: Dict[str, RAPID_BaseRegression]):
        """
        Args:
            models: A dictionary where keys are model names and 
                    values are fitted RAPID regression objects.
        """
        self.models = models
        self._validate_models()

    def _validate_models(self):
        for name, model in self.models.items():
            if not hasattr(model, 'performance_metrics_df') or model.performance_metrics_df is None:
                raise ValueError(f"Model '{name}' must be fitted before comparison.")

    def compare_performance(self) -> pd.DataFrame:
        """
        Combines performance metrics from all models into a single table.
        """
        dfs = []
        for name, model in self.models.items():
            temp_df = model.performance_metrics_df.copy()
            temp_df.columns = ['Metric', name]
            temp_df.set_index('Metric', inplace=True)
            dfs.append(temp_df)
        
        comparison_df = pd.concat(dfs, axis=1)
        return comparison_df

    def compare_coefficients(self) -> pd.DataFrame:
        """
        Combines summary/coefficient dataframes to compare effect sizes.
        """
        dfs = []
        for name, model in self.models.items():
            # Use the summary_df created during .fit()
            temp_df = model.summary_df.copy()
            
            # Clean column names (strip (uni)/(multi) suffixes)
            temp_df.columns = temp_df.columns.str.replace(r' \((uni|multi)\)', '', regex=True)
            
            # Keep only the variable and the effect estimate
            # Logic handles both 'Coefficient' (Linear) and 'OddsRatio' (Logistic)
            effect_col = 'OddsRatio' if 'OddsRatio' in temp_df.columns else 'Coefficient'
            
            temp_df = temp_df[['Variable', effect_col]]
            temp_df.columns = ['Variable', f"{name}_Estimate"]
            temp_df.set_index('Variable', inplace=True)
            dfs.append(temp_df)
            
        return pd.concat(dfs, axis=1)

    def report(self):
        """Prints a clean comparison report to the console."""
        print("=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        perf = self.compare_performance()
        print(perf.to_string())
        print("=" * 80)