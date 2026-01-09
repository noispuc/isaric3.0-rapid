import pandas as pd
import numpy as np
import warnings
# Importa apenas a sua classe principal, que agora é o pipeline completo
from survival_pipeline import RAPID_survival 

# Ignora warnings de aproximação do lifelines, comum em C-index e calibração
#warnings.filterwarnings('ignore', category=warnings.ApyWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- 1. Carregando os DataFrames ---
try:
    print("--- 1. Carregando DataFrames ---")
    df_model = pd.read_csv('df_model.csv') # Caso 1: Dados do modelo
    df_map = pd.read_csv('df_map.csv') # Caso 2: Dados MAP
    print(f"Datasets carregados com sucesso. Linhas no Caso 1: {len(df_model)}.")
except FileNotFoundError:
    print("ERRO: Certifique-se de que 'df_model.csv' e 'df_map.csv' estão no diretório de execução.")
    exit()
print("-" * 50)

# =================================================================
#                           USER CASE 1
# =================================================================
print("================== INICIANDO USER CASE 1 ==================")
duration_col_c1 = 'HospitalLengthStay_trunc'
event_col_c1 = 'HospitalDischargeCode_trunc_bin'
predictors_c1 = [
    'period', 'Idade_Agrupada2', 'ChronicHealthStatusName', 'obesity',
    'IsImmunossupression', 'IsSteroidsUse', 'IsSevereCopd', 'IsChfNyha',
    'cancer', 'ResourceIsRenalReplacementTherapy', 'ResourceIsVasopressors',
    'Vent_Resource'
]
target_time_c1 = 60.0 # Tempo alvo para ROC/Calibração (ex: 60 dias)

# --- 2. Criação e Execução do Pipeline ---
pipeline_c1 = RAPID_survival(
    data=df_model,
    duration_col=duration_col_c1,
    event_col=event_col_c1,
    predictors=predictors_c1
)

print("\n--- 2. FASE: FIT DO MODELO ---")
# O .fit() realiza o pré-processamento (one-hot encoding, limpeza de NaNs) e treina o modelo Cox.
pipeline_c1.fit()

print("\n--- 3. FASE: SUMMURY E DIAGNÓSTICO ---")
# O .summary() gera a tabela de HR, métricas de fit e todos os plots solicitados.
plots_c1 = [
    'forest_plot', 
    'schoenfeld_residuals', 
    'martingale_residuals', 
    'deviance_residuals',
    'roc_auc', 
    'calibration_plot'
]

pipeline_c1.summary(
    fit_measures=True, 
    plots=plots_c1, 
    target_time=target_time_c1
)

print("==================== USER CASE 1 CONCLUÍDO ====================")
print("\n" * 2)

# =================================================================
#                           USER CASE 2
# =================================================================
print("================== INICIANDO USER CASE 2 ==================")
# --- 1. Pré-processamento Manual Necessário (Cálculo de Duração) ---
df_cox_prep = df_map.copy()
df_cox_prep['dates_admdate'] = pd.to_datetime(df_cox_prep['dates_admdate'], errors='coerce')
df_cox_prep['outco_date'] = pd.to_datetime(df_cox_prep['outco_date'], errors='coerce')
df_cox_prep['duration_col'] = (df_cox_prep['outco_date'] - df_cox_prep['dates_admdate']).dt.days
df_cox_prep['outcome_binary'] = df_cox_prep['outco_binary_outcome'].map({
    "Death": 1, "Censored": 0, "Discharged": 0
})
print("Pré-processamento manual do Caso 2 (cálculo de duração) concluído.")

duration_col_c2 = 'duration_col'
event_col_c2 = 'outcome_binary'
predictors_c2 = [
    'demog_sex', 'demog_healthcare',
    'comor_hypertensi', 'comor_chrkidney', 'comor_liverdisease', 'comor_obesity', 
    'comor_chrkidney_stag', 'comor_liverdisease_type'
]
target_time_c2 = 12.0 # Tempo alvo para ROC/Calibração (ex: 12 dias)

# --- 2. Criação e Execução do Pipeline ---
pipeline_c2 = RAPID_survival(
    data=df_cox_prep, # Usando o df pré-processado
    duration_col=duration_col_c2,
    event_col=event_col_c2,
    predictors=predictors_c2
)

print("\n--- 2. FASE: FIT DO MODELO ---")
pipeline_c2.fit()

print("\n--- 3. FASE: SUMMURY E DIAGNÓSTICO ---")
# O .summary() para o segundo caso
plots_c2 = [
    'forest_plot', 
    'schoenfeld_residuals', 
    'martingale_residuals', 
    'deviance_residuals',
    'roc_auc', 
    'calibration_plot'
]

pipeline_c2.summary(
    fit_measures=True, 
    plots=plots_c2, 
    target_time=target_time_c2
)

print("==================== USER CASE 2 CONCLUÍDO ====================")