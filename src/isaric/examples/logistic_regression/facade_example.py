"""
Regressão logística — exemplo FACADE (camada iniciante)
"""
import pandas as pd
from isaric.modeling.pipeline_factory import RAPID_PipelineFactory

df = pd.read_csv("dados.csv")

factory = RAPID_PipelineFactory()
model = factory.create(
    "logistic",
    data=df,
    dependent_var="death",
    independent_vars=["age", "sex", "sofa_score"],
    family="binomial",
    link="logit",
)

model.fit(cross_val=True, n_splits=5)
model.summary(
    assumptions="all",
    performance="all",
    plots=["forest_plot", "roc_curve", "confusion_matrix"],
)
model.report()
