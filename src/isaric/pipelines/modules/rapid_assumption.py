import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import warnings

# High-performance statistical imports from your reqs
from lifelines.utils import concordance_index
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

class ModelAssumptionTester:
    """
    A model-agnostic assumption tester utilizing optimized backends.
    Calculates diagnostics for survival, regression, and classification models.
    """
    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        y_pred: Optional[np.ndarray] = None
    ):
        self.model = model
        self.X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.y = np.array(y).flatten()
        self.y_pred = self._get_predictions(y_pred)
        self.residuals = self.y - self.y_pred
        self.n, self.p = self.X.shape

    def _get_predictions(self, y_pred: Optional[np.ndarray]) -> np.ndarray:
        if y_pred is not None:
            return np.array(y_pred).flatten()
        
        # Priority: .predict() (sklearn/lifelines) -> .fittedvalues (statsmodels)
        if hasattr(self.model, 'predict'):
            return self.model.predict(self.X)
        if hasattr(self.model, 'fittedvalues'):
            return np.array(self.model.fittedvalues).flatten()
        
        raise ValueError("Model must have a .predict() method or .fittedvalues attribute.")

    def test_c_index(self, event: Optional[np.ndarray] = None) -> float:
        """O(n log n) Concordance Index calculation via lifelines."""
        # Defaults to all events observed (1) if event array is None
        e = event if event is not None else np.ones_like(self.y)
        return concordance_index(self.y, -self.y_pred, e)

    def test_normality(self) -> Dict[str, Any]:
        """Shapiro-Wilk test for normality of residuals."""
        res_sample = self.residuals
        if self.n > 5000:
            # Shapiro-Wilk is not valid for N > 5000 in scipy.
            # This will take a random sample of the data up to the maximum allowed limit.
            res_sample = np.random.choice(res_sample, 5000, replace=False)
        stat, p = shapiro(res_sample)
        return {"statistic": stat, "p_value": p, "is_normal": p > 0.05}

    def test_durbin_watson(self) -> float:
        """Durbin-Watson test for autocorrelation (Independence of errors)."""
        return durbin_watson(self.residuals)

    def test_vif(self) -> pd.DataFrame:
        """Variance Inflation Factor for multicollinearity."""
        # Constant is required for valid VIF calculation in statsmodels
        X_with_const = self.X.assign(constant=1.0)
        vif_df = pd.DataFrame()
        vif_df["feature"] = self.X.columns
        vif_df["VIF"] = [
            variance_inflation_factor(X_with_const.values, i) 
            for i in range(self.p)
        ]
        return vif_df

    def test_epv(self, event: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculates Events Per Variable (EPV).
        For survival/binary models: (Total Events) / (Number of Predictors)
        For continuous models: (Total N) / (Number of Predictors)
        """
        # If event is provided (survival/binary), use sum of events.
        # Otherwise, use total sample size (n) for linear regression EPV.
        if event is not None:
            n_events = np.sum(event)
            context = "Survival/Binary (Events)"
        else:
            # Check if y appears to be binary [0, 1]
            unique_y = np.unique(self.y)
            if len(unique_y) == 2:
                n_events = np.min([np.sum(self.y == val) for val in unique_y])
                context = "Classification (Minority Class)"
            else:
                n_events = self.n
                context = "Linear Regression (Total N)"

        # Avoid division by zero if p=0
        epv_value = n_events / self.p if self.p > 0 else np.inf
        
        # Standard thresholds: < 10 is high risk, 10-20 is moderate/caution.
        status = "Robust"
        if epv_value < 10:
            status = "High Risk (Overfitting Likely)"
        elif epv_value < 20:
            status = "Caution (Low Power)"

        return {
            "epv": round(epv_value, 2),
            "n_events": int(n_events),
            "n_predictors": self.p,
            "context": context,
            "status": status
        }