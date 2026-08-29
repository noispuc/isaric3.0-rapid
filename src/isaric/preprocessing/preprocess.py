"""
Step 2: Data Preprocessing - Prepare data for modelling and analysis.

This module provides the Preprocess class, which orchestrates all data
preprocessing techniques of the RAPID methodology.

The Preprocess class follows the same pattern as all RAPID classes:
- Configuration is set through class attributes
- Public execute() method orchestrates the configured techniques
- Private methods call functions from technique-specific modules
"""

import pandas as pd
from typing import Optional

from isaric.preprocessing.datasplitting import (
    parse_split_strategy,
    simple_random_split,
    stratified_split,
    temporal_split
)
from isaric.preprocessing.imputation import (
    parse_imputation_strategy,
    mice_imputation
)
from isaric.preprocessing.collinearity import (
    parse_collinearity_strategy,
    vif_analysis,
    pearson_correlation
)
from isaric.preprocessing.normalization import (
    parse_normalization_strategy,
    standardize,
    minmax_scale
)
from isaric.preprocessing.encoding import (
    parse_encoding_strategy,
    onehot_encode,
    label_encode,
    target_encode
)
from isaric.preprocessing.scaling import (
    parse_scaling_strategy,
    log_transform,
    boxcox_transform
)
from isaric.preprocessing.featureselection import (
    parse_selection_strategy,
    variance_threshold,
    lasso_selection,
    rfe_selection,
    filter_selection
)
from isaric.preprocessing.temporalencoding import (
    parse_temporal_strategy,
    duration_encode,
    cyclical_encode
)


class Preprocess:
    """
    Step 2: Data Preprocessing.

    Orchestrates data preprocessing techniques based on configured attributes.

    Attributes:
        data_splitting (str): Data splitting strategy.
            Options:
                - None (skip)
                - "split(test=0.2)"
                - "split(test=0.2,stratify=outcome)"
                - "split(test=0.2,method=temporal,date_col=admission_date)"
        imputation (str): Imputation strategy.
            Options:
                - None (skip)
                - "imputation(type=mice)"
        collinearity (str): Collinearity analysis strategy.
            Options:
                - None (skip)
                - "vif(threshold=5.0)"
                - "pearson(threshold=0.75)"
        normalization (str): Normalization strategy.
            Options:
                - None (skip)
                - "standardize"
                - "minmax"
        encoding (str): Encoding strategy.
            Options:
                - None (skip)
                - "onehot"
                - "label"
                - "target"
        scaling (str): Scaling strategy.
            Options:
                - None (skip)
                - "log"
                - "boxcox"
        feature_selection (str): Feature selection strategy.
            Options:
                - None (skip)
                - "variance(threshold=0.0)"
                - "lasso(n=10)"
                - "rfe(n=15)"
                - "filter(threshold=0.1)"
        temporal_encoding (bool): Temporal encoding activation.
            Options:
                - False (skip)
                - True (executes with default cyclical period=7)
    """

    def __init__(
        self,
        data_splitting: Optional[str] = None,
        imputation: Optional[str] = None,
        collinearity: Optional[str] = None,
        normalization: Optional[str] = None,
        encoding: Optional[str] = None,
        scaling: Optional[str] = None,
        feature_selection: Optional[str] = None,
        temporal_encoding: bool = False
    ):
        """
        Initialize the Preprocess class with configured techniques.

        Args:
            data_splitting: Strategy for data splitting.
            imputation: Strategy for imputation (MICE).
            collinearity: Strategy for collinearity analysis.
            normalization: Strategy for normalization.
            encoding: Strategy for encoding.
            scaling: Strategy for scaling.
            feature_selection: Strategy for feature selection.
            temporal_encoding: Enable temporal encoding.
        """
        self.data_splitting = data_splitting
        self.imputation = imputation
        self.collinearity = collinearity
        self.normalization = normalization
        self.encoding = encoding
        self.scaling = scaling
        self.feature_selection = feature_selection
        self.temporal_encoding = temporal_encoding

    # ======================================================================
    # PUBLIC METHOD
    # ======================================================================

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Execute all configured preprocessing techniques.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with configured preprocessing applied.
        """
        result = data.copy()

        if self.data_splitting:
            result = self._split_data(result)

        if self.imputation:
            result = self._impute(result)

        if self.collinearity:
            result = self._analyze_collinearity(result)

        if self.normalization:
            result = self._normalize(result)

        if self.encoding:
            result = self._encode(result)

        if self.scaling:
            result = self._scale(result)

        if self.feature_selection:
            result = self._select_features(result)

        if self.temporal_encoding:
            result = self._encode_temporal(result)

        return result

    # ======================================================================
    # PRIVATE METHODS - TECHNIQUES
    # ======================================================================

    def _split_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data splitting based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with split applied.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, test_size, target_or_date_col = parse_split_strategy(
            self.data_splitting
        )

        if method == "random":
            train_df, test_df = simple_random_split(data, test_size=test_size)
            return pd.concat([train_df, test_df], ignore_index=True)

        elif method == "stratified":
            train_df, test_df = stratified_split(
                data,
                target_col=target_or_date_col,
                test_size=test_size
            )
            return pd.concat([train_df, test_df], ignore_index=True)

        elif method == "temporal":
            train_df, test_df = temporal_split(
                data,
                date_col=target_or_date_col,
                test_size=test_size
            )
            return pd.concat([train_df, test_df], ignore_index=True)

        else:
            raise ValueError(f"Unknown split method: {method}")

    def _impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with missing values imputed.

        Raises:
            ValueError: If strategy is unknown.
        """
        params = parse_imputation_strategy(self.imputation)

        if params["type"] == "mice":
            return mice_imputation(
                data,
                n=params.get("n", 5),
                max_iter=params.get("max_iter", 10)
            )

        else:
            raise ValueError(f"Unknown imputation type: {params['type']}")

    def _analyze_collinearity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply collinearity analysis based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame without highly collinear features.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, threshold = parse_collinearity_strategy(self.collinearity)

        if method == "vif":
            return vif_analysis(data, threshold=threshold)

        elif method == "pearson":
            return pearson_correlation(data, threshold=threshold)

        else:
            raise ValueError(f"Unknown collinearity method: {method}")

    def _normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with normalized numeric columns.

        Raises:
            ValueError: If strategy is unknown.
        """
        method = parse_normalization_strategy(self.normalization)

        if method == "standardize":
            return standardize(data)

        elif method == "minmax":
            return minmax_scale(data)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def _encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with encoded categorical columns.

        Raises:
            ValueError: If strategy is unknown.
        """
        method = parse_encoding_strategy(self.encoding)

        if method == "onehot":
            return onehot_encode(data)

        elif method == "label":
            return label_encode(data)

        elif method == "target":
            # Requires target_col - needs to be passed in the strategy
            target_col = self.encoding.split("target_col=")[1].rstrip(")")
            return target_encode(data, target_col=target_col)

        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def _scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with scaled numeric columns.

        Raises:
            ValueError: If strategy is unknown.
        """
        method = parse_scaling_strategy(self.scaling)

        if method == "log":
            return log_transform(data)

        elif method == "boxcox":
            return boxcox_transform(data)

        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature selection based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with selected features.

        Raises:
            ValueError: If strategy is unknown.
        """
        method, parameter = parse_selection_strategy(self.feature_selection)

        if method == "variance":
            return variance_threshold(data, threshold=parameter)

        elif method == "lasso":
            target_col = self.feature_selection.split("target_col=")[1].rstrip(")")
            return lasso_selection(
                data,
                target_col=target_col,
                n_features=int(parameter)
            )

        elif method == "rfe":
            target_col = self.feature_selection.split("target_col=")[1].rstrip(")")
            return rfe_selection(
                data,
                target_col=target_col,
                n_features=int(parameter)
            )

        elif method == "filter":
            target_col = self.feature_selection.split("target_col=")[1].rstrip(")")
            return filter_selection(
                data,
                target_col=target_col,
                threshold=parameter
            )

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

    def _encode_temporal(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply temporal encoding based on configured strategy.

        Args:
            data: Input DataFrame.

        Returns:
            DataFrame with temporal variables encoded.

        Raises:
            ValueError: If strategy is unknown.
        """
        if isinstance(self.temporal_encoding, bool):
            # Apply default cyclical encoding (period=7)
            # This is a placeholder - actual implementation depends on data
            return data

        elif isinstance(self.temporal_encoding, str):
            method, params = parse_temporal_strategy(self.temporal_encoding)

            if method == "duration":
                return duration_encode(
                    data,
                    start_col=params["start"],
                    end_col=params["end"],
                    unit=params.get("unit", "days")
                )

            elif method == "cyclical":
                return cyclical_encode(
                    data,
                    column=params["column"],
                    period=params.get("period", 7)
                )

        raise ValueError(
            f"Unknown temporal encoding: {self.temporal_encoding}"
        )