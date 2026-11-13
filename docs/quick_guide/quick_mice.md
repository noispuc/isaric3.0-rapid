#  MICE Imputer

!!! abstract "TL;DR"
    **Assinatura:** `MICEImputer(n=5, max_iter=10, random_state=42)`
    **O que faz:** Imputação múltipla por equações encadeadas para dados missing.
    **Quando usar:** Datasets clínicos com valores ausentes em variáveis numéricas e categóricas.

---
This page provides a quickstart reference for the MICEImputer class, covering initialisation, fitting, and results retrieval.This page provides a quickstart reference for the MICEImputer class, covering initialisation, fitting, and results retrieval. For full parameter documentation, output details, and statistical methodology, see the MICE Imputer User Guide.

##  Quick Reference

| Method | Description | Example |
|--------|-------------|---------|
| `MICEImputer()` | Initialize imputer | `imputer = MICEImputer(n=5, max_iter=10, random_state=42)` |
| `fit()` | Run imputation | `imputer.fit(df, interest_vars=['age', 'sex', 'outcome'])` |
| `get_results()` | Retrieve outputs | `datasets, pooled, stats = imputer.get_results()` |

---

##  Parameters - MICEImputer()

| Parameter | Type | Default | Notes | Methodological Stage |
|-----------|------|---------|-------|----------------------|
| `n` | int | `5` | Number of imputed datasets | Preprocessing |
| `max_iter` | int | `10` | Iterations per imputation | Preprocessing |
| `random_state` | int | `None` | Seed for reproducibility | Preprocessing |
| `initial_strategy` | str | `'most_frequent'` | `'mean'`, `'median'`, `'most_frequent'`, `'constant'` | Preprocessing |
| `force_binary` | bool | `False` | Round dummies to 0/1 | Preprocessing |
| `cutoff` | float | `0.5` | Threshold for binarising | Preprocessing |
| `tie_strategy` | str | `'highest_probability'` | `'highest_probability'`, `'first'`, `'nan'`, `'force_class'` | Preprocessing |

---

##  Parameters - fit()

| Parameter | Type | Default | Description | Methodological Stage |
|-----------|------|---------|-------------|----------------------|
| `df` | DataFrame | required | Input data with missing values | Preprocessing |
| `interest_vars` | list | `None` | Variables to include in stats table | Evaluation |
| `show_results` | bool | `True` | Print stats table after fitting | Evaluation |

---

##  Main Outputs - get_results()

| Output | Type | Description |
|--------|------|-------------|
| `datasets` | list of DataFrame | `n` individually imputed datasets |
| `pooled` | DataFrame | Element-wise mean across all imputations |
| `stats` | DataFrame | Comparative statistics table |

---

##  stats_df Columns

| Column | Description |
|--------|-------------|
| `VARIABLE` | Variable name |
| `MISSINGS` | Count (%) of missing values |
| `BEFORE IMPUTATION` | Original distribution |
| `AFTER IMPUTATION` | Imputed distribution |
| `P-VALUE` | Mann-Whitney U or chi-squared test |
| `IMPUTATION VARIANCE` | Variance across imputations |

---

##  Minimal Example

```python
import pandas as pd
from mice_imputer import MICEImputer

# Load data
df = pd.read_csv("your_data.csv")

# Define variables of interest
interest_vars = ['age', 'sex', 'bmi', 'outcome']

# Initialize and run imputation
imputer = MICEImputer(
    n=5,
    max_iter=10,
    random_state=42,
    tie_strategy='highest_probability'
)

imputer.fit(df, interest_vars=interest_vars)

# Retrieve results
datasets, pooled, stats = imputer.get_results()

# Use individual datasets for downstream modeling
for ds in datasets:
    model.fit(ds[['age', 'sex', 'bmi']], ds['outcome'])

# Inspect statistics
print(stats[['VARIABLE', 'MISSINGS', 'P-VALUE', 'IMPUTATION VARIANCE']])
```

##  Tie-Breaking Strategies

| Strategy | Description |
|----------|-------------|
| `'highest_probability'` | Assign class with highest imputed value (default) |
| `'first'` | Assign first class in dummy order |
| `'nan'` | Leave as `NaN` |
| `'force_class'` | Assign specific class via `tie_force` |

---

##  Output Files

After `fit()`, a file `imputed_datasets.zip` is saved with `n` Excel files:

imputed_datasets.zip
imputed_dataset_1.xlsx
imputed_dataset_2.xlsx
...
imputed_dataset_n.xlsx


##  Quick Links

| Want to... | Go to... |
|------------|----------|
| Understand the theory? | **[MICE Tutorial](../user_guide/guide_mice.md)** |