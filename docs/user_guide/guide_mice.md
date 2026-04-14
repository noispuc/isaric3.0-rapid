# MICEImputer
*Multiple Imputation by Chained Equations — Code Documentation*

---

## Concept

Missing data is a common challenge in clinical and epidemiological research. MICE (Multiple Imputation by Chained Equations) addresses this by generating multiple plausible completed datasets, running statistical models on each, and pooling the results — preserving uncertainty about the missing values rather than replacing them with a single guess.

### How MICE Works

MICE operates iteratively. For each variable with missing data, it fits a regression model using all other variables as predictors, then imputes the missing values by drawing from the resulting predictive distribution. This cycle repeats for a configurable number of rounds (`max_iter`), after which one completed dataset is produced. The whole process is then repeated `n` times with different random seeds, yielding `n` independently imputed datasets.

> **Key idea:** Running `n` independent imputations gives you `n` plausible versions of the complete dataset. Downstream analyses (e.g. regression models) are run on each dataset separately, and the results are combined using Rubin's Rules to get valid point estimates and standard errors that properly account for imputation uncertainty.

### Handling Categorical Variables

sklearn's `IterativeImputer` only handles numeric data. `MICEImputer` extends it to categorical variables by one-hot encoding them before imputation and reconstructing the original categories afterwards. The reconstruction uses configurable thresholds (`cutoff`) and tie-breaking strategies (`tie_strategy`) to convert imputed dummy probabilities back into discrete class labels.

---

## Class Overview

`MICEImputer` wraps sklearn's `IterativeImputer` to support mixed-type datasets. It handles the full imputation pipeline: encoding, running `n` imputations, pooling, statistical analysis, and saving outputs.

```python
from mice_imputer import MICEImputer

imputer = MICEImputer(
    n=5,
    max_iter=10,
    random_state=42
)
imputer.fit(df, interest_vars=['age', 'sex', 'outcome'])
datasets, pooled, stats = imputer.get_results()
```

---

## Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n` | `5` | Number of independent imputations to produce. More imputations reduce Monte Carlo error but increase runtime. |
| `max_iter` | `10` | Maximum imputation rounds per dataset, passed directly to `IterativeImputer`. |
| `initial_strategy` | `'most_frequent'` | Initialisation strategy for missing values before the iterative loop: `'mean'`, `'median'`, `'most_frequent'`, or `'constant'`. |
| `random_state` | `None` | Base seed for reproducibility. `n` unique seeds are derived from it. If `None`, a random base seed is drawn each run. |
| `prefix_sep` | `'!'` | Separator for one-hot encoded dummy columns (e.g. `'sex!Male'`). Must not appear in any existing column name. |
| `force_binary` | `False` | If `True`, rounds imputed dummy values to hard 0/1 using the cutoff threshold instead of leaving them as probabilities. |
| `cutoff` | `0.5` | Global decision threshold for binarising dummies when `force_binary=True`. Values >= cutoff become 1, below become 0. |
| `cutoff_map` | `None` | Per-variable threshold overrides (dict mapping variable name to float). Variables absent from this dict fall back to `cutoff`. |
| `tie_strategy` | `'highest_probability'` | Tie-breaking strategy when multiple dummies are >= cutoff: `'highest_probability'`, `'first'`, `'nan'`, or `'force_class'`. |
| `tie_force` | `None` | Class to assign when `tie_strategy='force_class'`. Accepts a string (same class for all variables) or a dict mapping variable to class. |
| `imputer_kwargs_extra` | `None` | Additional keyword arguments forwarded directly to `IterativeImputer` (e.g. `estimator`, `skip_complete`). |

---

## Public API

### `fit(df, interest_vars=None, show_results=True)`

Runs the full MICE pipeline on the input dataframe. This is the main entry point.

**Parameters**

| Parameter | Description |
|-----------|-------------|
| `df` | `pd.DataFrame` with the input data. May contain missing values and both numerical and categorical columns. |
| `interest_vars` | List of column names to include in the statistical analysis table. Columns without missing values are used as predictors only and will not appear in the stats table. If `None`, every column is used. |
| `show_results` | If `True` (default), prints the statistical analysis table to stdout after fitting. |

**Returns** `self`, enabling method chaining.

---

### `get_results()`

Returns the three output objects produced by `fit()`. Raises `RuntimeError` if called before `fit()`.

**Returns**

| Object | Description |
|--------|-------------|
| `imputed_datasets_` | list of `pd.DataFrame` — the `n` individually imputed datasets, one per seed. |
| `pooled_df_` | `pd.DataFrame` — element-wise mean across all `n` imputed arrays, with categoricals reconstructed. |
| `stats_df_` | `pd.DataFrame` — comparative statistical analysis table (see Output DataFrames section). |

---

## Output DataFrames

### `imputed_datasets_`

A Python list of `n` DataFrames. Each is a fully completed copy of the original data with the same column names, index, and dtypes. The datasets differ from one another because each was produced with a distinct random seed, introducing the stochastic variation that is the point of multiple imputation.

> **Recommended use:** Fit your downstream model (e.g. logistic regression) separately on each of the `n` datasets. Combine the resulting coefficient estimates and standard errors using Rubin's Rules to obtain a single pooled estimate that correctly propagates imputation uncertainty.

### `pooled_df_`

A single DataFrame with the same shape and columns as the original. Missing cells are filled with the element-wise mean across all `n` imputed arrays, then categorical columns are reconstructed from the averaged dummy probabilities. Non-missing cells are identical to the original input.

> **Note:** Averaging dummy probabilities before argmax is a rougher pooling strategy than Rubin's Rules. `pooled_df_` is convenient for quick exploratory analysis, but for inferential modelling the individual datasets in `imputed_datasets_` should be preferred.

### `stats_df_`

A summary DataFrame with one row per variable that had at least one missing value. Variables with no missings are silently excluded. The columns are:

| Column | Description |
|--------|-------------|
| `VARIABLE` | Name of the variable. |
| `MISSINGS` | Count and percentage of missing values in the original data, formatted as `'N (X.XX%)'`. |
| `BEFORE IMPUTATION` | Descriptive statistics on the original (incomplete) column: median (Q1, Q3) for continuous variables; count and % of positives for binary/categorical variables. |
| `AFTER IMPUTATION` | Same statistics computed on the corresponding column of `pooled_df_`. |
| `P-VALUE` | Mann-Whitney U test for continuous variables; chi-squared test for categorical variables. Tests whether the distribution changed significantly after imputation. |
| `IMPUTATION VARIANCE` | Average variance across the `n` imputations at the originally-missing positions. Higher values indicate more disagreement among imputations and greater uncertainty about those cells. |

---

## Example

### Import and Instantiate

```python
import pandas as pd
from mice_imputer import MICEImputer

url = "https://raw.githubusercontent.com/ISARICResearch/VERTEX/main/.../df_map.csv"
df = pd.read_csv(url)
```

### Define Variables of Interest

```python
interest_vars = [
    'demog_age', 'demog_sex',
    'comor_hypertensi', 'comor_diabetes_yn',
    'vital_highesttem_c', 'labs_creatinine_mgdl'
]
```

### Run Imputation

```python
imputer = MICEImputer(
    n=10,
    max_iter=10,
    random_state=42,
    tie_strategy='highest_probability'
)

imputer.fit(df, interest_vars=interest_vars, show_results=True)
```

### Retrieve Results

```python
datasets, pooled, stats = imputer.get_results()

# Inspect pooled dataset
print(pooled.head())

# Inspect statistical summary
print(stats[['VARIABLE', 'MISSINGS', 'P-VALUE', 'IMPUTATION VARIANCE']])

# Use individual datasets for Rubin's Rules
for ds in datasets:
    model.fit(ds[predictors], ds[outcome])
```

---

## Tie-Breaking Strategies

When reconstructing categorical columns from one-hot dummies, ties arise when either multiple dummies exceed the cutoff, or no dummy exceeds it. The `tie_strategy` parameter controls what happens in these cases:

| Strategy | Description |
|----------|-------------|
| `'highest_probability'` | (default) Among all tied dummy columns, assign the class with the highest raw imputed value. |
| `'first'` | Assign the first class in the dummy column order. |
| `'nan'` | Assign `NaN`, preserving the ambiguity rather than forcing a class. |
| `'force_class'` | Assign a specific pre-defined class given by `tie_force` (string for all variables, or dict mapping variable to class). |

---

## Reproducibility

Set `random_state` to any integer to guarantee identical results across runs. The imputer derives `n` independent seeds from this base seed, one per imputation. When `random_state=None`, a random base seed is drawn; the seeds actually used are stored in `imputer._seeds` after fitting, so results can always be recovered.

---

## Output Files

After `fit()` completes, a file named `imputed_datasets.zip` is saved in the working directory. It contains `n` Excel files:

```
imputed_datasets.zip
  imputed_dataset_1.xlsx
  imputed_dataset_2.xlsx
  ...
  imputed_dataset_n.xlsx
```

Each Excel file corresponds to one imputed dataset with the row index preserved. These files are the primary deliverable for passing to external analysis tools or sharing with collaborators.

---

## Advantages

- **Handles mixed types:** numerical and categorical variables are supported in a single call.
- **Multiple datasets:** produces `n` datasets that capture imputation uncertainty, enabling valid inference via Rubin's Rules.
- **Configurable:** tie-breaking, cutoffs, initialisation strategy, and imputer internals are all accessible through constructor parameters.
- **Statistical audit:** the `stats_df_` table lets you immediately see whether imputation meaningfully shifted each variable's distribution.
- **Reproducible:** fixed `random_state` guarantees identical outputs across runs, with seeds stored in `_seeds` for inspection.

## Limitations

- **Categorical pooling:** `pooled_df_` averages dummy probabilities before argmax, which is an approximation. For rigorous inference, use `imputed_datasets_` with Rubin's Rules.
- **Assumes MAR:** like all MICE implementations, this class assumes data is Missing At Random. Results may be biased under MNAR mechanisms.
- **Scale:** `IterativeImputer` fits a separate model per variable per iteration. Very wide datasets with many categorical variables can be slow.
- **Stats table:** the p-value column compares before vs. after distributions, not a formal test of imputation quality. Treat it as a diagnostic, not a decision rule.

---

## References

- scikit-learn IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html