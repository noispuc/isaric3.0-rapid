"""
Cox PH — exemplo FACADE (camada iniciante)
"""
import pandas as pd
from isaric.modeling.pipeline_factory import RAPID_PipelineFactory

df = pd.read_csv("dados.csv")

factory = RAPID_PipelineFactory()
model = factory.create(
    "survival",
    data=df,
    duration_var="los_days",
    dependent_var="death",
    independent_vars=["age", "sex", "sofa_score"],
)

model.fit(penalizer=0.1, cross_val=True, n_splits=5)
model.summary(
    assumptions=True,
    performance=True,
    plots=["forest_plot", "schoenfeld", "martingale", "brier_score", "calibration"],
    target_time=30,
)
model.report()
