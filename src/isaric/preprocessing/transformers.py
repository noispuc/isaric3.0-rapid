"""
Sklearn-compatible preprocessing transformers for the RAPID pipeline.

These classes provide fit/transform separation so that preprocessing
statistics (mean, std, imputation values, category mappings) are learned
exclusively from the training set and then applied to the test set.

Usage:
    scaler = RAPIDStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class RAPIDStandardScaler:
    """Stateful standard scaler: fit on train, transform on test."""

    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
        self._scaler = None
        self._fit_columns: List[str] = []

    def fit(self, X: pd.DataFrame) -> "RAPIDStandardScaler":
        from sklearn.preprocessing import StandardScaler
        cols = self.columns if self.columns is not None else list(X.select_dtypes(include="number").columns)
        self._fit_columns = cols
        self._scaler = StandardScaler()
        self._scaler.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._scaler is None:
            raise RuntimeError("Call fit() before transform().")
        result = X.copy()
        result[self._fit_columns] = self._scaler.transform(X[self._fit_columns])
        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def mean_(self) -> np.ndarray:
        return self._scaler.mean_ if self._scaler is not None else None

    @property
    def scale_(self) -> np.ndarray:
        return self._scaler.scale_ if self._scaler is not None else None


class RAPIDOneHotEncoder:
    """Stateful one-hot encoder: fit on train, transform on test."""

    def __init__(self, drop_first: bool = True):
        self.drop_first = drop_first
        self._categories: dict = {}
        self._cat_columns: List[str] = []

    def fit(self, X: pd.DataFrame) -> "RAPIDOneHotEncoder":
        self._cat_columns = list(X.select_dtypes(include=["object", "category"]).columns)
        for col in self._cat_columns:
            self._categories[col] = sorted(X[col].dropna().unique().tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._categories and self._cat_columns:
            raise RuntimeError("Call fit() before transform().")
        result = X.copy()
        for col in self._cat_columns:
            cats = self._categories.get(col, [])
            result[col] = pd.Categorical(result[col], categories=cats)
        dummies = pd.get_dummies(result, columns=self._cat_columns, drop_first=self.drop_first)
        return dummies

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


class RAPIDImputer:
    """Stateful MICE imputer: fit on train, transform on test."""

    def __init__(
        self,
        n: int = 5,
        max_iter: int = 10,
        initial_strategy: str = "most_frequent",
        random_state: int = 42,
    ):
        self.n = n
        self.max_iter = max_iter
        self.initial_strategy = initial_strategy
        self.random_state = random_state
        self._imputers: list = []

    def fit(self, X: pd.DataFrame) -> "RAPIDImputer":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self._imputers = []
        for _ in range(self.n):
            imp = IterativeImputer(
                max_iter=self.max_iter,
                initial_strategy=self.initial_strategy,
                random_state=self.random_state,
            )
            imp.fit(X)
            self._imputers.append(imp)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._imputers:
            raise RuntimeError("Call fit() before transform().")
        imputed = np.mean(
            [imp.transform(X) for imp in self._imputers], axis=0
        )
        return pd.DataFrame(imputed, columns=X.columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
