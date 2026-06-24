import pandas as pd
from typing import Dict
from isaric.modeling.base_model import RAPID_BasePipeline

class RAPID_ModelComparator:
    def __init__(self, models: Dict[str, RAPID_BasePipeline]):
        """
        Args:
            models: A dictionary where keys are model names and 
                    values are fitted RAPID pipeline objects.
        """
        self.models = models
        self._validate_models()

    def _validate_models(self):
        for name, model in self.models.items():
            if not hasattr(model, 'performance_metrics_df') or model.performance_metrics_df is None:
                raise ValueError(f"Model '{name}' must be fitted before comparison.")
            if not hasattr(model, 'summary_df') or model.summary_df is None:
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
        
        return pd.concat(dfs, axis=1)

    def compare_summary(self) -> pd.DataFrame:
            """
            Combines summary dataframes from all models into a single table.
            """
            dfs = []
            for name, model in self.models.items():
                temp_df = model.summary_df.copy()
                temp_df.set_index('Variable', inplace=True)
                temp_df.columns = [f"{name}_{col}" for col in temp_df.columns]
                dfs.append(temp_df)

            return pd.concat(dfs, axis=1)

    def report(self):
        """Prints a clean comparison report to the console."""
        print("=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        print("\nPERFORMANCE METRICS")
        print("=" * 80)
        print(self.compare_performance().to_string())
        print("=" * 80)
        print("\nSUMMARY")
        print("=" * 80)
        print(self.compare_summary().to_string())
        print("=" * 80)