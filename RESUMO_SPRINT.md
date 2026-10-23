# Resumo do Sprint — branch `revarch_predictive`

**Data:** 2026-10-23 | **Branch:** `revarch_predictive` | **Suíte:** 107 passed, 4 skipped, 0 failed

---

**Objetivo:** corrigir o pacote `isaric3.0-rapid` para reprodução fiel do paper Delpino et al. (mortalidade por dengue pediátrica) via API pública `RAPID.create → fit → summary`, sem workarounds externos.

**Issues corrigidas (6 de 8):**
Issues 3–8 foram resolvidas em 13 commits com testes dedicados: (3) `class_weight` configurável; (4) métricas no hold-out com threshold configurável; (5) `fit(balancing=, recalibrate=True)` com slope/intercept de calibração; (6) transformadores `RAPIDStandardScaler/OneHotEncoder/Imputer` sem leakage; (7) threshold O(n), `summary(short)` e lazy-import catboost; (8) `grid_search` sem `AttributeError`, recalibração isotônica real, ICs bootstrap nos gráficos de calibração.

**Métricas reproduzidas vs. referência:**
AUC obtido = **0.867** vs. paper **0.866** (Δ = +0.001); calibration slope ≈ 1.006 (paper 1.006); threshold fixado em 0.155 como no paper. Modelo: `logistic_l2`, `undersample`, `recalibrate=True`, `random_seed=42`.

**Pendente — bootstrap de métricas do hold-out (issue 5 parcial):**
`model.metrics` não contém ICs para AUC, F1 etc. — a infraestrutura (`bootstrap.py`) existe mas não está conectada ao `fit()`. Impacto: o IC reportado no paper (AUC 0.860–0.871) não é reproduzível pela API pública ainda.

**Não abordado:** issues 1 e 2 (pré-existentes, fora do escopo definido para este sprint).
