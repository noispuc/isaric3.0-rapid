import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import warnings

from lifelines.utils import concordance_index
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, chi2


class ModelAssumptionTester:
    """
    Model-agnostic assumption tester for survival, regression, and classification models.

    Parameters
    ----------
    model : fitted model object
        Must have .fittedvalues or .predict().
    X : array-like or pd.DataFrame
        Feature matrix used to fit the model.
    y : array-like
        True target values.
    y_pred : array-like or None
        Pre-computed predictions. If None, derived from model.
    """

    def __init__(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        y_pred: Optional[np.ndarray] = None,
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
        if hasattr(self.model, 'fittedvalues'):
            return np.array(self.model.fittedvalues).flatten()
        if hasattr(self.model, 'predict'):
            return self.model.predict(self.X)
        raise ValueError("Model must have a .predict() method or .fittedvalues attribute.")

    def test_c_index(self, event: Optional[np.ndarray] = None) -> float:
        """Concordance Index via lifelines. Defaults to all events observed if event is None."""
        e = event if event is not None else np.ones_like(self.y)
        return concordance_index(self.y, -self.y_pred, e)

    def test_normality(self) -> Dict[str, Any]:
        """Shapiro-Wilk test for normality of residuals."""
        res_sample = self.residuals
        if self.n > 5000:
            res_sample = np.random.choice(res_sample, 5000, replace=False)
        stat, p = shapiro(res_sample)
        return {"statistic": stat, "p_value": p}

    def test_durbin_watson(self) -> float:
        """Durbin-Watson test for autocorrelation of errors."""
        return durbin_watson(self.residuals)

    def test_vif(self) -> pd.DataFrame:
        """Variance Inflation Factor — detects multicollinearity."""
        X = self.X
        if not any(col.lower() in ("intercept", "const") for col in X.columns):
            X = X.assign(constant=1.0)
        vif_df = pd.DataFrame({
            "feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })
        return vif_df

    def test_epv(self, event: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Events Per Variable (EPV).
        For survival/binary: (Total Events) / (Number of Predictors).
        For continuous: (Total N) / (Number of Predictors).
        """
        if event is not None:
            n_events = np.sum(event)
            context = "Survival/Binary (Events)"
        else:
            unique_y = np.unique(self.y)
            if len(unique_y) == 2:
                n_events = np.min([np.sum(self.y == val) for val in unique_y])
                context = "Classification (Minority Class)"
            else:
                n_events = self.n
                context = "Linear Regression (Total N)"

        epv_value = n_events / self.p if self.p > 0 else np.inf

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
            "status": status,
        }

    def test_cooks_distance(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Cook's Distance influence diagnostic using QR decomposition.

        Args:
            threshold: Custom threshold for influential points (default: 4/n).
        """
        if self.n <= self.p + 1:
            raise ValueError(f"Need n > p+1: n={self.n}, p={self.p}")

        X_mat = np.column_stack([np.ones(self.n), self.X.values])
        Q, _ = np.linalg.qr(X_mat)
        leverage = np.sum(Q ** 2, axis=1)

        dof = self.n - self.p - 1
        mse = np.sum(self.residuals ** 2) / dof
        std_res = self.residuals / np.sqrt(mse * np.maximum(1 - leverage, 1e-10))
        cooks_d = (std_res ** 2 / (self.p + 1)) * (leverage / np.maximum(1 - leverage, 1e-10))

        thresh = threshold if threshold is not None else 4 / self.n
        influential_idx = np.where(cooks_d > thresh)[0]

        return {
            "cooks_distance": cooks_d,
            "max_distance": float(np.max(cooks_d)),
            "influential_indices": influential_idx.tolist(),
            "threshold": float(thresh),
            "n_influential": len(influential_idx),
        }

    @staticmethod
    def likelihood_ratio_test(
        null_ll: float, alt_ll: float, null_dof: int, alt_dof: int
    ) -> Dict[str, Any]:
        """
        Likelihood Ratio Test for comparing nested models.

        Args:
            null_ll: Log-likelihood of the simpler (null) model.
            alt_ll: Log-likelihood of the complex (alternative) model.
            null_dof: Degrees of freedom of the null model.
            alt_dof: Degrees of freedom of the alternative model.
        """
        lr_stat = 2 * (alt_ll - null_ll)
        dof_diff = abs(alt_dof - null_dof)
        if lr_stat < 0:
            lr_stat = 0.0
        p_value = chi2.sf(lr_stat, dof_diff)
        return {
            "log_likelihood_diff": lr_stat,
            "dof_diff": dof_diff,
            "p_value": float(p_value),
        }
