"""
Abstract class representing a generic pipeline.
Designed to be inherited from by all other pipeline classes.
"""

from abc import ABC, abstractmethod

class RAPID_Pipeline(ABC):

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess_data():
        pass

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def summary():
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------

    @abstractmethod
    def _data_cleaning():
        pass
    
    @abstractmethod
    def _preprocessing():
        pass

    @abstractmethod
    def _modeling():
        pass

    @abstractmethod
    def _model_evaluation():
        pass

    @abstractmethod
    def _validation():
        pass

    @abstractmethod
    def _visualization():
        pass