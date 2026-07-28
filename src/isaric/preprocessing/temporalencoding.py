import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def encode_temporal(df, date_column, reference_date=None):
    """
    Description:
        Encodes temporal variables as time differences from a reference date.

    Args:
        df (pandas.DataFrame): Input dataset.
        date_column (str): Name of the date column.
        reference_date (str or datetime): Reference date for encoding.

    Returns:
        pandas.DataFrame: Dataset with encoded temporal feature.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    ref = pd.to_datetime(reference_date) if reference_date else df[date_column].min()
    df[date_column + "_days"] = (df[date_column] - ref).dt.days
    return df.drop(columns=[date_column])


class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Description:
        Transformer sklearn-compatível que codifica uma variável temporal
        periódica (mês, semana epidemiológica, etc.) em duas features
        contínuas (seno e cosseno), preservando a proximidade circular entre
        os extremos do período (ex.: dezembro e janeiro ficam próximos).
        Relevante para capturar sazonalidade em arboviroses como dengue.

        Sem estado a ser aprendido no treino (fit é no-op), portanto seguro
        para uso dentro de um sklearn.Pipeline sem risco de vazamento entre
        treino e teste.

    Args:
        period (float): Período do ciclo (12 para mês, 52 para semana
            epidemiológica).
        source (str): Como interpretar a(s) coluna(s) de entrada:
            - 'raw': valores numéricos já no domínio do período (ex.: 1-12).
            - 'date': valores de data, dos quais o mês (1-12) é extraído.
            - 'epiweek': strings/inteiros no formato AAAASS (ex.: '201920'),
              dos quais os 2 últimos dígitos são extraídos como semana.

    Notes:
        Valores ausentes na coluna de origem propagam como NaN nas features
        seno/cosseno. Posicione este encoder antes do imputador dentro do
        Pipeline para que o NaN residual seja tratado normalmente.
    """

    def __init__(self, period: float, source: str = "raw"):
        self.period = period
        self.source = source

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raw = self._extract_raw_values(X)
        angle = 2 * np.pi * raw / self.period
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None):
        base = "cyclical" if input_features is None else "_".join(map(str, input_features))
        return np.array([f"{base}_sin", f"{base}_cos"])

    def _extract_raw_values(self, X):
        series = self._as_series(X)

        if self.source == "raw":
            return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

        if self.source == "date":
            return pd.to_datetime(series, errors="coerce").dt.month.to_numpy(dtype=float)

        if self.source == "epiweek":
            weeks = series.astype(str).str.strip().str[-2:]
            return pd.to_numeric(weeks, errors="coerce").to_numpy(dtype=float)

        raise ValueError(f"source inválido: '{self.source}'. Use 'raw', 'date' ou 'epiweek'.")

    @staticmethod
    def _as_series(X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    "CyclicalFeatureEncoder espera exatamente uma coluna de entrada, "
                    f"recebeu {X.shape[1]}."
                )
            return X.iloc[:, 0]
        if isinstance(X, pd.Series):
            return X
        array = np.asarray(X)
        if array.ndim == 2:
            if array.shape[1] != 1:
                raise ValueError(
                    "CyclicalFeatureEncoder espera exatamente uma coluna de entrada, "
                    f"recebeu {array.shape[1]}."
                )
            array = array.ravel()
        return pd.Series(array)
