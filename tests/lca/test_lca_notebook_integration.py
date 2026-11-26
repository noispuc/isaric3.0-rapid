"""
Notebook integration tests for RAPID v0.2.0 LCA development notebook.

Verifies:
- Original reference notebook is preserved untouched with its outputs.
- Adapted local notebook (notebooks/validation/lca_development_local.ipynb)
  removes remote pip install and Colab dependency, uses pathlib.Path,
  has clean empty outputs on disk, and defines independent objects
  model_no_bootstrap and model_bootstrap.
- Full end-to-end execution of the adapted notebook via nbclient in fast mode.
"""

from pathlib import Path
import nbclient
import nbformat
import pytest


def test_original_reference_notebook_is_preserved(repo_root: Path):
    """Ensure RAPIDv0_2_0_LCA_Clustering_sampleSINAN.ipynb exists and keeps its outputs."""
    orig_path = repo_root / "RAPIDv0_2_0_LCA_Clustering_sampleSINAN.ipynb"
    assert orig_path.exists()

    nb = nbformat.read(orig_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    cells_with_outputs = [c for c in code_cells if c.get("outputs")]
    assert len(cells_with_outputs) >= 5, "Original notebook outputs must not be stripped"


def test_adapted_local_notebook_structure_and_conventions(repo_root: Path):
    """Verify structural requirements of notebooks/validation/lca_development_local.ipynb."""
    nb_path = repo_root / "notebooks" / "validation" / "lca_development_local.ipynb"
    assert nb_path.exists()

    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 6

    all_code = "\n".join(c.source for c in code_cells)

    # 1. No remote pip install or mandatory google.colab import
    assert "pip install git+https://github.com/noispuc/isaric3.0-rapid.git" not in all_code
    assert "from google.colab import files" not in all_code

    # 2. Uses pathlib.Path and checks isaric.__file__
    assert "from pathlib import Path" in all_code
    assert "isa.__file__" in all_code

    # 3. Separate modeling objects
    assert "model_no_bootstrap" in all_code
    assert "model_bootstrap" in all_code

    # 4. Clean outputs on disk
    for idx, cell in enumerate(code_cells):
        assert cell.get("outputs") == [], f"Code cell {idx} should not have saved outputs on disk"


def test_adapted_local_notebook_executes_via_nbclient(repo_root: Path, monkeypatch):
    """Execute notebooks/validation/lca_development_local.ipynb in-memory via nbclient."""
    nb_path = repo_root / "notebooks" / "validation" / "lca_development_local.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    monkeypatch.setenv("RAPID_LCA_FAST_TEST", "1")
    monkeypatch.setenv("RAPID_LCA_FORCE_SYNTHETIC", "1")

    client = nbclient.NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    executed_nb = client.execute()

    code_cells = [c for c in executed_nb.cells if c.cell_type == "code"]
    all_stdout = "\n".join(
        out.get("text", "")
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "stream" and out.get("name") == "stdout"
    )

    assert "Validação confirmada: isaric carregado diretamente do código local!" in all_stdout
    assert "model_no_bootstrap treinado" in all_stdout
    assert "model_bootstrap treinado" in all_stdout
    assert "Independência confirmada: model_no_bootstrap permaneceu intacto" in all_stdout
