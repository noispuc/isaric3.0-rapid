"""
Abstract class representing a generic pipeline.
Designed to be inherited from by all other pipeline classes.
"""

from abc import ABC, abstractmethod

class RAPID_BasePipeline(ABC):


    @abstractmethod
    def _preprocess_data():
        """
        Should call the _data_cleaning() and _preprocessing() methods.
        """
        pass

    # ------------------------------------------------------------------
    # PUBLIC METHODS
    # ------------------------------------------------------------------

    @abstractmethod
    def fit():
        """
        Should call _modeling(). 
        Subsequently, should call _model_evaluation() and extract relevant assumptions tests and fit metrics.
        """ 
        pass

    @abstractmethod
    def summary():
        """
        Should call visualizaiton().
        """
        pass

    @abstractmethod
    def report():
        """
        Should display all metrics and all plots without filters (full summary).
        """
        pass

    @abstractmethod
    def validate():
        """
        Should call modules from validation/ (bootstrap, external, subgroup, etc.).
        """
        pass

    # ------------------------------------------------------------------
    # PRIVATE METHODS (FOLLOWING THE STANDARD ISARIC PIPELINE STRUCTURE)
    # ------------------------------------------------------------------

    @abstractmethod
    def _data_cleaning():
        """
        Format the data such that can be utilized for the statistical model after the preprocessing stage. 
        For instance, in a regresison, should remove null values.
        """
        pass
    
    @abstractmethod
    def _preprocessing():
        """
        Should use the rapid_preprocess util to extract relevant matrices and generate
        a formula for the model.
        """
        pass

    @abstractmethod
    def _modeling():
        """
        Should fit the model and store it in self.fit_model()
        """
        pass

    @abstractmethod
    def _model_evaluation():
        """
        If applicable, should check assumption tests.
        Should evaluate fit/performance metrics. Should run cross validation if applicable. 
        Performance metrics and assumption tests should each be stored in a separate dataframe.
        """
        pass

    @abstractmethod
    def _validation():
        """
        Should provide a structure for comparing the metrics generated in the _model_evaluation() stage.
        For multiple models stored in the same pipeline.
        """
        pass

    @abstractmethod
    def _visualization():
        """
        Should optionally display assumption test results, performance metrics, as well as any and all relevant plots.
        """
        pass