#  Examples Gallery

Welcome to the **RAPID Examples Gallery**. Here you'll find Jupyter notebooks demonstrating real-world applications of the RAPID methodology.

Each example includes:
- Data loading and preprocessing
- Model initialization and fitting
- Results interpretation
- Diagnostic plots

---

##  Available Examples

| Example | Description | Download | View | Run Online |
|---------|-------------|----------|------|------------|
| **Logistic Regression** | Binary outcome prediction (mortality, readmission) | [ `.ipynb`](logistic_example.ipynb){:download} | [ View](logistic_example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noispuc/isaric3.0-rapid/blob/restructure/docs/examples/logistic_example.ipynb){:target="_blank"} |
| **GLM** | Generalized Linear Models with different families | [ `.ipynb`](glm_example.ipynb){:download} | [ View](glm_example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noispuc/isaric3.0-rapid/blob/restructure/docs/examples/glm_example.ipynb){:target="_blank"} |
| **Survival Analysis** | Time-to-event analysis with Cox PH | [ `.ipynb`](survival_example.ipynb){:download} | [ View](survival_example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noispuc/isaric3.0-rapid/blob/restructure/docs/examples/survival_example.ipynb){:target="_blank"} |
| **MICE Imputation** | Multiple imputation for missing data | [ `.ipynb`](mice_example.ipynb){:download} | [ View](mice_example.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/noispuc/isaric3.0-rapid/blob/restructure/docs/examples/mice_example.ipynb){:target="_blank"} |

---

##  Quick Start with Examples

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/noispuc/isaric3.0-rapid.git
cd rapid-pipeline

# Install RAPID
pip install -e .

# Launch Jupyter
jupyter notebook docs/examples/
```

### Option 2: Run on Google Colab
Click the Colab badge above to open any notebook directly in Google Colab. Then run:

```bash
!pip install git+https://github.com/noispuc/isaric3.0-rapid.git
```

### Option 3: View Static Rendering
Click View to see the notebook rendered directly in your browser (no execution).

##  Questions?

- **Email:** [data@isaric.org](mailto:data@isaric.org)
- **GitHub Issues:** [noispuc/isaric3.0-rapid](https://github.com/noispuc/isaric3.0-rapid/issues)
