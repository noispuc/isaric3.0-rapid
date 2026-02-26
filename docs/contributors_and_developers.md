# 🏗️ For Contributors and Developers

Thank you for your interest in contributing to the RAPID Pipeline development!

 To maintain consistency, cientific integrity and performance of our tools, across biostatistical analyses and machine learning models, we follow a strict architectural pattern. All contributors must adhere to the standards outlined below.

---

## The RAPID_Pipeline Base Class

To ensure seamless integration with our ecosystem, **all classes must inherit from the `RAPID_Pipeline` base class**. This inheritance guarantees that every model or analysis tool shares a common interface, making the library predictable for end-users.

```python
from rapid.base import RAPID_Pipeline

class MyNewModel(RAPID_Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

```

---

## 🧩 Abstract Classes and Structure

We utilize **Abstract Base Classes (ABCs)** to define the skeleton of our algorithms. This enforcement ensures that developers implement the necessary logic before the code can even be instantiated.

> All new pipelines must inherit from `RAPID_Pipeline` available at as it implement a modular structure that ensures consistency across different analytical tasks.

* **Mandatory Implementation:** You are required to override specific abstract methods defined in the parent classes.
* **Validation:** Use the `@abstractmethod` decorator from the `abc` module when proposing new base structures.
* **Internal Logic:** Always use the internal helper methods provided by `RAPID_Pipeline` for data validation and logging to keep the scientific output consistent.

---

## 🛠️ Standardization of User Interaction

To provide a unified experience for biostatisticians and researchers, every model in this library must implement the following three core methods:

1. **`.fit(data, **params)`**: The primary method for training or fitting the statistical model. It must handle data validation internally.
2. **`.summary()`**: Must return a structured overview of the results (e.g., coefficients, p-values, confidence intervals). This is the "scientific" view of the model.
3. **`.report()`**: Generates a comprehensive, human-readable output (often in Markdown or LaTeX) suitable for clinical or biological research documentation.

---

## 📏 Standardization of Arguments and Parameters

Consistency in naming conventions is non-negotiable. This prevents confusion when switching between different types of analyses (e.g., switching from a T-test to a Random Forest).

* **Naming Convention:** Use `snake_case` for all arguments.
* **Input Data:** Always use `X` for features/independent variables and `y` for labels/dependent variables in ML contexts. For pure biostatistics, use `data` and `group`.
* **Hyperparameters:** All model parameters must be defined in the `__init__` method with sensible default values based on current biostatistical literature.
* **Return Types:** Ensure that methods returning statistical values use standard Python types or NumPy arrays to maintain compatibility with the rest of the pipeline.

---

## ✅ Contribution Pull Request (PR) Checklist

Before submitting a Pull Request, ensure that:

* [ ] Your class inherits from `RAPID_Pipeline`.
* [ ] All abstract methods are fully implemented.
* [ ] The `fit`, `summary`, and `report` methods follow the library's output standards.
* [ ] Type hints are included for all arguments and return values.
* [ ] Documentation strings (Docstrings) follow the NumPy/SciPy format.

---
