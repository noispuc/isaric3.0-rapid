"""
Cox PH — exemplo MODULAR (camada avançada)
"""
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

from isaric.preprocessing.formulaprocessor import RapidPreprocessor
from isaric.visualization.forestplots import ForestPlot
from isaric.visualization.residualplots import ResidualPlots
from isaric.visualization.calibrationplot import CalibrationPlot
from isaric.visualization.brierscoreplot import BrierScorePlot

df = pd.read_csv("dados.csv")
duration_var = "los_days"
event_var = "death"
independent_vars = ["age", "sex", "sofa_score"]

# 1. Pré-processamento
y, X, _ = RapidPreprocessor.prepare_data(
    df=df,
    target_cols=[duration_var, event_var],
    predictor_cols=independent_vars,
    intercept=False,
)
model_data = pd.concat([y, X], axis=1)

# 2. Modelagem
cox = CoxPHFitter(penalizer=0.1)
cox.fit(model_data, duration_col=duration_var, event_col=event_var)
print(f"C-Index: {cox.concordance_index_:.4f}")

# 3. Visualização — Forest Plot (Hazard Ratios)
summary = cox.summary.copy()
summary["HazardRatio"] = np.exp(summary["coef"])
summary["LowerCI"] = np.exp(summary["coef"] - 1.96 * summary["se(coef)"])
summary["UpperCI"] = np.exp(summary["coef"] + 1.96 * summary["se(coef)"])
summary = summary.reset_index().rename(columns={summary.index.name or "covariate": "Variable"})

ForestPlot.plot(
    df=summary,
    label_col="Variable",
    effect_col="HazardRatio",
    lower_col="LowerCI",
    upper_col="UpperCI",
    title="Forest Plot — Hazard Ratios (Cox)",
    null_value=1.0,
    log_scale=True,
).show()

# 4. Resíduos de Schoenfeld
res_sch = cox.compute_residuals(model_data, "schoenfeld")
for col in res_sch.columns:
    ResidualPlots.schoenfeld_plot(
        residuals=res_sch[col].values,
        times=res_sch.index.values,
        covariate_name=col,
    ).show()

# 5. Calibração (t=30 dias)
target_time = 30
CalibrationPlot.survival_calibration(
    fitted_model=cox,
    df=model_data,
    duration_col=duration_var,
    event_col=event_var,
    t=target_time,
).show()
