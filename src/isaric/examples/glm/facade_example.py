"""
GLM gaussiano — exemplo FACADE (camada iniciante)

O usuário cria a pipeline via factory e chama fit/summary/report/validate
sem precisar conhecer os módulos internos.
"""
import pandas as pd
from isaric.modeling.pipeline_factory import RAPID_PipelineFactory

df = pd.read_csv("dados.csv")

factory = RAPID_PipelineFactory()
model = factory.create(
    "glm",
    data=df,
    dependent_var="outcome",
    independent_vars=["age", "sex", "bmi"],
    family="gaussian",
    link="identity",
)

model.fit(cross_val=True, n_splits=5)
model.summary(
    assumptions="all",
    performance="all",
    plots=["forest_plot", "residuals_vs_fitted", "qq_plot"],
)
model.report()
