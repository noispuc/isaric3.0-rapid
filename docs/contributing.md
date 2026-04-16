# 🏗️ Contributing to RAPID

Thank you for your interest in contributing to the **RAPID Pipeline**!

To maintain consistency, scientific integrity, and performance across biostatistical analyses and machine learning models, we follow a strict architectural pattern. All contributors must adhere to the standards outlined below.

---

## 🚀 Development Environment Setup

### 1. Clone and Install

```bash
git clone git clone https://github.com/noispuc/isaric3.0-rapid.git
cd rapid-pipeline
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## 2. Project Structure

src/isaric/
├── base/           # Abstract base classes
├── pipelines/      # Model implementations
├── preprocessing/  # Data cleaning and imputation
├── modeling/       # Statistical models
├── evaluation/     # Metrics and diagnostics
├── validation/     # Cross-validation and bootstrapping
└── visualization/  # Plotting functions

## 🧬 The RAPID_Pipeline Base Class

To ensure seamless integration, **all model classes must inherit from `RAPID_Pipeline`**. This guarantees a common interface across all analytical tools.

```python
from isaric.base import RAPID_Pipeline

class MyNewModel(RAPID_Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

---

## 🧩 Abstract Methods (Required)

All new pipelines must implement the following core methods:

| Method | Purpose | Return |
|--------|---------|--------|
| `fit()` | Train the model with preprocessing | `self` |
| `summary()` | Generate diagnostics and results | `None` (prints/displays) |

---

## Example Template
```python
class MyNewModel(RAPID_Pipeline):
    def __init__(self, data, param1=default1, param2=default2):
        super().__init__(data=data)
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, **kwargs):
        # 1. Validate input data
        # 2. Preprocess
        # 3. Train model
        # 4. Store results in self.summary_df, self.metrics_df, etc.
        return self
    
    def summary(self, plots=None, **kwargs):
        # 1. Print results table
        # 2. Generate requested plots
        pass
```

## 🏭 Factory Integration
All models must be registered with the RAPID_PipelineFactory to enable the unified create() interface:

```python
# In your model file
from isaric.pipelines.factory import RAPID_PipelineFactory

@RAPID_PipelineFactory.register("my_model")
class MyNewModel(RAPID_Pipeline):
    ...
```

This allows users to instantiate your model with:
```python
model = factory.create("my_model", data=df, param1=value)
```

## 📏 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Variables/Functions | `snake_case` | `dependent_var`, `calculate_auc()` |
| Classes | `PascalCase` | `LogisticRegression`, `SurvivalCox` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_THRESHOLD` |
| Private methods | `_leading_underscore` | `_validate_data()` |

---

## 📤 Return Types and Attributes

All models should expose results via standardized attributes:

| Attribute | Type | Content |
|-----------|------|---------|
| `summary_df` | `pd.DataFrame` | Coefficients, CI, p-values |
| `performance_metrics_df` | `pd.DataFrame` | Model fit metrics |
| `assumption_metrics_df` | `pd.DataFrame` | Diagnostic tests |
| `vif_df` | `pd.DataFrame` | Multicollinearity (if applicable) |

---

## 📝 Documentation

Update the following files when adding a new model:

| File | What to Add |
|------|-------------|
| `docs/user_guide/guide_*.md` | Full tutorial with theory |
| `docs/quick_guide/quick_*.md` | Quick reference table |
| `docs/index.md` | Link in "Choose Your Path" |
| `mkdocs.yml` | Nav entry under User Guide and Quick Guide |

---

## ✅ Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Class inherits from `RAPID_Pipeline`
- [ ] Registered with `RAPID_PipelineFactory`
- [ ] `fit()` and `summary()` methods implemented
- [ ] Standard attributes populated (`summary_df`, `performance_metrics_df`)
- [ ] Type hints included for all arguments and return values
- [ ] Docstrings follow [PEP 257](https://peps.python.org/pep-0257/)
- [ ] Code follows [PEP 8](https://peps.python.org/pep-0008/)
- [ ] Tests added in `tests/`
- [ ] Documentation updated (User Guide + Quick Guide)

---

## 💬 Questions?

- **Email:** [data@isaric.org](mailto:data@isaric.org)
- **GitHub Issues:** [ISARICResearch/rapid-pipeline](https://github.com/ISARICResearch)