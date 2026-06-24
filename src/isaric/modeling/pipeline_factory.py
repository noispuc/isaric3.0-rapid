from typing import Type, Dict
from isaric.modeling.base_model import RAPID_BasePipeline
from isaric.modeling.glm import RAPID_GLM
from isaric.modeling.logistic_regression import RAPID_LogisticRegression
from isaric.modeling.survival import RAPID_SurvivalCox
from isaric.modeling.LCA import RAPID_PhenotypeLCA

class RAPID_PipelineFactory:
    def __init__(self):
        self._registry: Dict[str, Type[RAPID_BasePipeline]] = {
            "glm": RAPID_GLM,
            "logistic": RAPID_LogisticRegression,
            "survival": RAPID_SurvivalCox,
            "lca": RAPID_PhenotypeLCA,
        }

    def register(self, name: str, pipeline_cls: Type[RAPID_BasePipeline]):
        """
        Register a custom pipeline class under a given name.

        Args:
            name: The identifier to register the pipeline under.
            pipeline_cls: The pipeline class to register (must inherit from RAPID_BasePipeline).
        """
        if not issubclass(pipeline_cls, RAPID_BasePipeline):
            raise ValueError(f"'{name}' must be a subclass of RAPID_BasePipeline.")
        self._registry[name] = pipeline_cls

    def create(self, name: str, **kwargs) -> RAPID_BasePipeline:
        """
        Instantiate a pipeline by name.

        Args:
            name: The identifier of the pipeline. Built-in options: 'linear', 'logistic'.
            **kwargs: Arguments passed to the pipeline constructor.

        Returns:
            An instance of the requested pipeline.
        """
        if name.lower() not in self._registry:
            raise ValueError(f"No pipeline registered under '{name}'. Available: {list(self._registry.keys())}")
        return self._registry[name.lower()](**kwargs)

    def available(self) -> list:
        """Returns a list of all registered pipeline names."""
        return list(self._registry.keys())