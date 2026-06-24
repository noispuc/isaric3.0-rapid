"""
Regressão logística — exemplo MODULAR (camada avançada)
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from isaric.preprocessing.formulaprocessor import RapidPreprocessor
from isaric.modelevaluation.assumptiontester import ModelAssumptionTester
from isaric.visualization.forestplots import ForestPlot
from isaric.visualization.rocplot import ROCPlot
from isaric.visualization.confusionmatrixplot import ConfusionMatrixPlot

df = pd.read_csv("dados.csv")
dependent_var = "death"
independent_vars = ["age", "sex", "sofa_score"]

# 1. Pré-processamento
y, X, _ = RapidPreprocessor.prepare_data(
    df=df,
    target_cols=[dependent_var],
    predictor_cols=independent_vars,
    intercept=True,
)

# 2. Modelagem
family = sm.families.Binomial(link=sm.families.links.Logit())
fitted_model = sm.GLM(endog=y, exog=X, family=family).fit()
print(fitted_model.summary())

# 3. Avaliação de pressupostos
tester = ModelAssumptionTester(model=fitted_model, X=X, y=y, y_pred=fitted_model.fittedvalues)
print("VIF:\n", tester.test_vif())
print("EPV:", tester.test_epv())

# 4. Visualização
y_pred = fitted_model.fittedvalues
y_class = (y_pred >= 0.5).astype(int)

fpr, tpr, _ = roc_curve(y, y_pred)
auc = roc_auc_score(y, y_pred)
ROCPlot.plot(fpr=fpr, tpr=tpr, auc=auc, title="ROC — Regressão Logística").show()

cm = confusion_matrix(y, y_class)
ConfusionMatrixPlot.plot(confusion_matrix=cm, class_names=["Vivo", "Óbito"]).show()

summary = fitted_model.summary2().tables[1]
or_df = pd.DataFrame({
    "Variable": summary.index,
    "OddsRatio": np.exp(summary["Coef."]),
    "Lower CI": np.exp(summary["[0.025"]),
    "Upper CI": np.exp(summary["0.975]"]),
})
ForestPlot.plot(
    df=or_df,
    label_col="Variable",
    effect_col="OddsRatio",
    lower_col="Lower CI",
    upper_col="Upper CI",
    title="Forest Plot — Odds Ratios",
    null_value=1.0,
    log_scale=True,
).show()
