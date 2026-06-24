"""
GLM gaussiano — exemplo MODULAR (camada avançada)

O usuário compõe as etapas manualmente importando cada módulo.
"""
import pandas as pd
from isaric.preprocessing.formulaprocessor import RapidPreprocessor
from isaric.modelevaluation.assumptiontester import ModelAssumptionTester
from isaric.visualization.forestplots import ForestPlot
from isaric.visualization.residualplots import ResidualPlots
import statsmodels.api as sm

df = pd.read_csv("dados.csv")
dependent_var = "outcome"
independent_vars = ["age", "sex", "bmi"]

# 1. Pré-processamento
y, X, _ = RapidPreprocessor.prepare_data(
    df=df,
    target_cols=[dependent_var],
    predictor_cols=independent_vars,
    intercept=True,
)

# 2. Modelagem
family = sm.families.Gaussian(link=sm.families.links.Identity())
fitted_model = sm.GLM(endog=y, exog=X, family=family).fit()
print(fitted_model.summary())

# 3. Avaliação de pressupostos
tester = ModelAssumptionTester(model=fitted_model, X=X, y=y, y_pred=fitted_model.fittedvalues)
print("VIF:\n", tester.test_vif())
print("Durbin-Watson:", tester.test_durbin_watson())
print("Shapiro-Wilk:", tester.test_normality())

# 4. Visualização
ForestPlot.plot(
    df=pd.DataFrame({
        "Variable": X.columns,
        "Coefficient": fitted_model.params[1:],
        "Lower CI": fitted_model.conf_int()[0][1:],
        "Upper CI": fitted_model.conf_int()[1][1:],
    }),
    label_col="Variable",
    effect_col="Coefficient",
    lower_col="Lower CI",
    upper_col="Upper CI",
    title="Forest Plot — GLM Gaussiano",
).show()

ResidualPlots.residuals_vs_fitted(
    residuals=fitted_model.resid_response,
    fitted_values=fitted_model.fittedvalues,
    title="Resíduos vs Ajustados",
).show()

ResidualPlots.qq_plot(
    residuals=fitted_model.resid_response,
    title="Q-Q Plot dos Resíduos",
).show()
