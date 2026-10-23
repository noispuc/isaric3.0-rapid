# RAPID Issues — branch `revarch_predictive`

Rastreamento das issues identificadas para permitir reprodução fiel do paper Delpino et al. (mortalidade por dengue em crianças) via `RAPID.create → fit → summary`.

**Ambiente de teste:** `conda run -n rapid python -m pytest tests/ -q`
**Status final da suíte:** 107 passed, 4 skipped (catboost não instalado), 0 failed.

---

## Issues corrigidas neste sprint

### Issue 3 — `class_weight='balanced'` hardcoded em todos os classificadores sklearn

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO |
| **Commit** | `edffac6` |
| **Arquivo** | `modeling/predictive.py`, `modeling/treebased.py` |
| **Problema** | `class_weight='balanced'` era hardcoded nos 9 classificadores sklearn, impossibilitando treinar sem balanceamento por peso. |
| **Solução** | Parâmetro `class_weight` exposto nas factory functions com default `None`; flui até o modelo sklearn. |
| **Testes** | `tests/test_issue3_class_weight.py` (17 passed, 2 skipped) |

---

### Issue 4 — `_calculate_metrics` avaliava no conjunto de treino com limiar fixo 0.5

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO |
| **Commit** | `7d0baa3` |
| **Arquivo** | `rapid.py`, `modeling/predictive.py`, `modeling/treebased.py` |
| **Problema** | Todas as métricas eram computadas sobre `self.X` / `self.y` (treino) com `threshold=0.5`, produzindo métricas otimistas e sensibilidade ≈ 0 em desfechos raros. |
| **Solução** | Split estratificado 70/30 no `create()` (`test_size`, `random_seed` configuráveis); `_calculate_metrics` avalia em `X_test`/`y_test`; limiar configurável via `fit(threshold=)` (float, `"f1"`, `"youden"`). |
| **Testes** | `tests/test_issue4_6_holdout.py` (21 passed) |

---

### Issue 5 — Balanceamento, recalibração e bootstrap de métricas do teste ausentes do pipeline

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO (exceto bootstrap de métricas — ver nota) |
| **Commits** | `df3096a`, `aac9bfb` |
| **Arquivo** | `rapid.py`, `validation/external.py` |
| **Problema** | Undersampling/SMOTE e recalibração logística não existiam no `fit()`; slope/intercept de calibração não eram calculados automaticamente. |
| **Solução** | `fit(balancing='undersample'/'smote')` usa `imbalanced-learn`; `fit(recalibrate=True)` aplica Platt scaling pós-treino; `metrics['calibration_slope']` e `metrics['calibration_intercept']` adicionados. Recalibração isotônica implementada em `validation/external.py`. |
| **Nota** | Bootstrap de métricas do hold-out (AUC CI, etc.) **não implementado** neste sprint — bootstrap de coeficientes já existia (`bootstrap_coefficients`). |
| **Testes** | `tests/test_issue5_balancing_recalibration.py` (13 passed), `tests/test_issue8b_isotonic.py` (5 passed) |

---

### Issue 6 — Transformadores de pré-processamento sem separação fit/transform

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO |
| **Commit** | `7d0baa3` |
| **Arquivo** | `preprocessing/transformers.py` (novo) |
| **Problema** | `standardize`, `onehot_encode`, `mice_imputation` chamavam `fit_transform()` no DataFrame inteiro, vazando estatísticas do teste para o treino. |
| **Solução** | Novas classes sklearn-compatíveis: `RAPIDStandardScaler`, `RAPIDOneHotEncoder`, `RAPIDImputer` — com `fit()` e `transform()` separados; fit obrigatoriamente feito apenas em `X_train`. Exportadas via `isaric.__init__`. |
| **Testes** | `tests/test_issue4_6_holdout.py` (testes de leakage incluídos) |

---

### Issue 7 — Performance e bugs de importação

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO |
| **Commits** | `64750bb`, `6a27e10`, `af7bef3` |
| **Sub-issues** | 7a) `select_classification_threshold` era O(n²) → vetorizado O(n); 7b) `summary(table_format='short')` quebrava em modelos predictive; 7c) `import isaric` falhava sem catboost |
| **Testes** | `tests/test_issue7a_threshold_vectorized.py`, `tests/test_issue7b_summary_short.py`, `tests/test_issue7c_import_no_catboost.py` |

---

### Issue 8 — Confiabilidade do grid search, recalibração e ICs

| Campo | Detalhe |
|---|---|
| **Status** | ✅ FEITO |
| **Commits** | `cc97f31`, `aac9bfb`, `c3a2d4d` |
| **Sub-issues** | 8a) `fit(grid_search=True)` levantava `AttributeError` em todos os modelos (atributos não inicializados); 8b) recalibração isotônica ignorava `method='isotonic'` e sempre fazia Platt; 8c) ICs dos gráficos de calibração eram ±10% fixos sem base estatística |
| **Solução** | 8a) `getattr(self, 'ATTR', None)` como fallback em todos os `best_params.get()`; 8b) branch isotônico real com `sklearn.isotonic.IsotonicRegression`; 8c) `calibration_curve_with_ci()` com bootstrap percentil (n=200) em `modelevaluation/calibration.py` |
| **Testes** | `tests/test_issue8a_gridsearch.py` (10 passed), `tests/test_issue8b_isotonic.py` (5 passed), `tests/test_issue8c_bootstrap_ci.py` (6 passed) |

---

## Issues pendentes (fora do escopo deste sprint)

### Issue 5 (parcial) — Bootstrap de métricas do hold-out

| Campo | Detalhe |
|---|---|
| **Status** | ⏳ PENDENTE |
| **Descrição** | `fit()` não gera ICs bootstrapados para AUC, F1, calibration slope etc. do conjunto de teste. A infraestrutura existe (`modelevaluation/bootstrap.py`), mas não está conectada ao pipeline `fit()`. |
| **Impacto** | `model.metrics` não contém `auc_ci_lower` / `auc_ci_upper`; valores do paper (ex. AUC 0.860–0.871) não são reproduzíveis diretamente pela API pública. |

### Issues 1 e 2 — Não abordadas neste sprint

Estas issues pré-existiam ao sprint mas estavam fora do escopo definido (reprodução do paper Delpino et al.). Não há documentação sobre seu conteúdo neste repositório.

---

## Métricas de reprodução (notebook `dengue-rapid-repro/predictive_dengue_usecase.ipynb`)

| Métrica | Paper (Delpino et al.) | Reproduzido | Δ |
|---|---|---|---|
| AUC | 0.866 (0.860–0.871) | 0.867 | +0.001 |
| Calibration slope | 1.006 | ~1.006 | ≈ 0 |
| Sensitivity | 0.359 | — | — |
| Specificity | 0.975 | — | — |
| Threshold | 0.155 | 0.155 (fixo) | 0 |

**Modelo:** `logistic_l2`, `balancing='undersample'`, `threshold=0.155`, `recalibrate=True`, `random_seed=42`.
