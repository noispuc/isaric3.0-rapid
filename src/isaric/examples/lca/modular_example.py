"""
LCA (Latent Class Analysis) — exemplo MODULAR (camada avançada)
"""
import pandas as pd
import numpy as np
from stepmix.stepmix import StepMix
from sklearn.base import clone

from isaric.visualization.lcaplots import LCAPlots
from isaric.modelevaluation.assumptiontester import ModelAssumptionTester

df = pd.read_csv("dados.csv")
measurement_vars = [c for c in df.columns if c.startswith("symptom_")]

df = df.dropna(subset=measurement_vars)
X = df[measurement_vars]
y = df["death"] if "death" in df.columns else None

# 1. Grid search manual sobre k
results = []
models = {}
for k in range(2, 8):
    m = StepMix(
        n_components=k,
        measurement="bernoulli",
        structural="bernoulli" if y is not None else None,
        random_state=42,
        verbose=0,
        progress_bar=0,
    )
    m.fit(X, y)
    n = X.shape[0]
    avg_ll = m.score(X, y)
    npar = m.n_parameters
    aic = -2 * avg_ll * n + 2 * npar
    bic = -2 * avg_ll * n + npar * np.log(n)
    results.append({"k": k, "AIC": aic, "BIC": bic})
    models[k] = m

grid_df = pd.DataFrame(results)
print(grid_df)
LCAPlots.plot_model_selection(grid_df.rename(columns={"k": "n_clusters"})).show()

# 2. Selecionar k=4 e descrever perfis
best_k = 4
best_model = models[best_k]

posteriors = best_model.predict_proba(X)
profiles = pd.DataFrame(
    (posteriors[:, :, None] * X.values[:, None, :]).sum(axis=0) / posteriors.sum(axis=0)[:, None],
    columns=measurement_vars,
    index=[f"Classe_{i+1}" for i in range(best_k)],
)
LCAPlots.plot_radar_profiles(profiles).show()
LCAPlots.plot_conditional_probs_line(profiles).show()

clusters = pd.Series(best_model.predict(X), index=df.index)
LCAPlots.plot_clusters(clusters).show()

# 3. Likelihood ratio test entre k=3 e k=4
ll3 = models[3].score(X, y) * X.shape[0]
ll4 = models[4].score(X, y) * X.shape[0]
lrt = ModelAssumptionTester.likelihood_ratio_test(ll3, ll4, models[3].n_parameters, models[4].n_parameters)
print("LRT k=3 vs k=4:", lrt)
