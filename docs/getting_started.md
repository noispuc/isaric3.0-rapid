# 🚀 Getting Started with `RAPID_pipeline`

The `RAPID_pipeline` is designed as a **modular and extensible structure** for implementing various Data Science models (Statistical, Survival Analysis, Machine Learning, etc.).

The current implementation focuses on the **Cox Proportional Hazards (Cox PH) model** for survival analysis, providing automated preprocessing, model fitting, and a comprehensive suite of diagnostic outputs.

Its core goal is to automate data preprocessing, model fitting, and the generation of publication-ready tables and diagnostic plots in a single, reusable object.

## 1\. Core Principles and Modularity

The pipeline is built on a simple three-phase methodology, which is universal for all models you integrate, ensuring high modularity and extensibility:

| Phase | Method | Role in the Pipeline |
| :--- | :--- | :--- |
| **Phase 1: Preprocessing** | `.preprocess_data()` | Handles data cleaning, missing value removal, and feature transformation (e.g., one-hot encoding). |
| **Phase 2: Training** | `.fit()` | Selects the model type (e.g., Cox PH, Logistic Regression, Random Forest) and performs the training procedure. |
| **Phase 3: Output/Diagnostics** | `.summary()` | Generates performance metrics, fit measures (e.g., AIC, C-Index), and produces a user-specified list of diagnostic plots. |

## 2\. Installation and Requirements
- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- Virtual Environment

To use the current **Survival Analysis** module, ensure the necessary that you have insdtalled Python in a 3.10+ version and also has [pip](https://pip.pypa.io/en/stable/). An virtual environment(venv) is recommended, all the required Python packages are listeded in [requirements.txt](https://github.com/noispuc/isaric3.0-rapid/tree/main) and the the venv can be set up with the following commands:

```bash
python -m venv .rapid
source .rapid/bin/activate  # ou .rapid\Scripts\activate no Windows
pip install -r requirements.txt
```
### 3\. Examples
Want to know how to instantiate and run a full survival analysis by defining the problem's scope (Duration, Event, Predictors) and letting the pipeline handle the rest?
* **[Go to: Running The Ppeline](running_the_pipeline.md)**
