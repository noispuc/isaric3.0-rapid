# ⚙️ Installation

This guide covers how to install the **ISARIC RAPID** package for different use cases.

---

## 📋 Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 or higher | [Download Python](https://www.python.org/downloads/) |
| pip | Latest | Included with Python 3.10+ |
| Virtual Environment | Recommended | `venv` or `conda` |

---

## 🚀 Quick Install (Recommended for Most Users)

Install the latest version directly from GitHub:

```bash
pip install git+https://github.com/noispuc/isaric3.0-rapid.git
```

This will install the `isaric` package and all required dependencies automatically.

!!! success "Verify Installation"
```python
    import isaric
    print(isaric.__version__)
```

---

## 🛠️ Install from Source (Developers & Contributors)

If you plan to modify the code or contribute to RAPID:

**1. Clone the Repository**
```bash
git clone git clone https://github.com/noispuc/isaric3.0-rapid.git
cd rapid-pipeline
```

**2. Create and Activate Virtual Environment**
=== "Windows"
```bash
python -m venv .venv
.venv\Scripts\activate
```

=== "macOS / Linux"
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install in Editable Mode**
```bash
pip install -e .
```
The `-e` flag installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

---

## ✅ Verify Installation
After installation, verify that RAPID is working correctly:
```python
from isaric.pipelines.factory import RAPID_PipelineFactory
print("RAPID installed successfully!")
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `pip: command not found` | Install pip or use `python -m pip` |
| `ModuleNotFoundError: isaric` | Ensure virtual environment is activated |
| Permission denied | Use `pip install --user` or virtual environment |

---

## 📚 Next Steps

- **[⚡ Quickstart Guide](quickstart.md)** – Run your first analysis.
- **[📖 User Guide](../user_guide/guide_logistic.md)** – Learn statistical methods.
- **[🤝 Contributing](../contributing.md)** – Set up development environment.