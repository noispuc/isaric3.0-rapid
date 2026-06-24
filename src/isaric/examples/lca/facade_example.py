"""
LCA (Latent Class Analysis) — exemplo FACADE (camada iniciante)
"""
import pandas as pd
from isaric.modeling.pipeline_factory import RAPID_PipelineFactory

df = pd.read_csv("dados.csv")

measurement_vars = [c for c in df.columns if c.startswith("symptom_")]

factory = RAPID_PipelineFactory()
model = factory.create(
    "lca",
    data=df,
    measurement_vars=measurement_vars,
    structural_var="death",
)

# Grid search automático para encontrar o melhor k
model.fit(cluster_range=range(2, 8))
model.summary()          # mostra métricas de seleção (AIC, BIC, entropia)
model.summary(k=4)       # descreve a solução com k=4 classes
model.report()
