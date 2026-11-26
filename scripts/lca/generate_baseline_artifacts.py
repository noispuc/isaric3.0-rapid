"""
Generate structured baseline artifacts in artifacts/lca_baseline/ for RAPID v0.2.0 LCA.

Usage:
    .venv/bin/python scripts/lca/generate_baseline_artifacts.py
    .venv/bin/python scripts/lca/generate_baseline_artifacts.py --clinical-data /path/to/DenguePos_child_Sample150.feather

Or configure the optional clinical dataset path via environment variable:
    RAPID_LCA_DATA_PATH=/path/to/DenguePos_child_Sample150.feather .venv/bin/python scripts/lca/generate_baseline_artifacts.py

Notes:
- Does not depend on any hardcoded machine-specific absolute paths.
- Clearly separates synthetic dataset runs from optional clinical dataset runs.
- Never stores individual clinical records or serialized models with patient data.
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isaric as isa
from tests.lca.synthetic_data import LCA_SYMPTOM_VARS, generate_synthetic_lca_data


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def _git_out(args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def resolve_optional_clinical_path(cli_path: Optional[str] = None) -> Optional[Path]:
    """
    Resolve an optional external clinical dataset path without hardcoded user paths.

    Priority:
    1. --clinical-data CLI argument
    2. RAPID_LCA_DATA_PATH environment variable
    3. Relative paths inside repository (data/DenguePos_child_Sample150.feather or ./DenguePos_child_Sample150.feather)
    4. ~/Downloads/DenguePos_child_Sample150.feather (if present)
    """
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    env_path = os.environ.get("RAPID_LCA_DATA_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(
        [
            REPO_ROOT / "data" / "DenguePos_child_Sample150.feather",
            REPO_ROOT / "DenguePos_child_Sample150.feather",
            Path.home() / "Downloads" / "DenguePos_child_Sample150.feather",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def run_dataset_scenarios(df: pd.DataFrame, dataset_label: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    timings = {}

    # Scenario A: No Bootstrap
    t0 = time.perf_counter()
    model_no_bootstrap = isa.RAPID.create(
        data=df,
        model="lca",
        n_clusters=4,
        measurement_vars=LCA_SYMPTOM_VARS,
        random_seed=42,
    )
    model_no_bootstrap.fit(
        bootstrap=False,
        param_grid={"n_clusters": [1, 2, 3, 4]},
    )
    timings["scenario_a_no_bootstrap_sec"] = round(time.perf_counter() - t0, 4)

    model_no_bootstrap.grid_results.to_csv(
        out_dir / "grid_results_no_bootstrap.csv", index=False
    )
    with open(out_dir / "model_info_no_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(model_no_bootstrap.model_info, f, indent=2, default=_json_default)

    # Scenario B: With Bootstrap (n_iterations=6)
    t0 = time.perf_counter()
    model_bootstrap = isa.RAPID.create(
        data=df,
        model="lca",
        n_clusters=4,
        measurement_vars=LCA_SYMPTOM_VARS,
        random_seed=42,
    )
    model_bootstrap.fit(
        bootstrap=True,
        n_iterations=6,
        param_grid={"n_clusters": [1, 2, 3, 4]},
    )
    timings["scenario_b_bootstrap_6iter_sec"] = round(time.perf_counter() - t0, 4)

    model_bootstrap.grid_results.to_csv(
        out_dir / "grid_results_bootstrap.csv", index=False
    )
    if model_bootstrap.lrt_results.get("ci_bootstrap") is not None:
        model_bootstrap.lrt_results["ci_bootstrap"].to_csv(
            out_dir / "ci_bootstrap.csv", index=False
        )
    if model_bootstrap.lrt_results.get("selection_rate") is not None:
        model_bootstrap.lrt_results["selection_rate"].to_csv(
            out_dir / "selection_rate.csv", index=False
        )
    if model_bootstrap.lrt_results.get("blrt") is not None:
        model_bootstrap.lrt_results["blrt"].to_csv(
            out_dir / "blrt_results.csv", index=False
        )
    with open(out_dir / "model_info_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump(model_bootstrap.model_info, f, indent=2, default=_json_default)

    # Generate plot evidence for all plots in plots_map
    stale_ci_png = plots_dir / "bootstrap_metrics_ci.png"
    if stale_ci_png.exists():
        stale_ci_png.unlink()

    plot_inventory = {}
    for plot_name, plot_fn in model_bootstrap.plots_map.items():
        fig_plotly = plot_fn(backend="plotly")
        fig_mpl = plot_fn(backend="matplotlib")
        mpl_axes_count = len(fig_mpl.axes)
        mpl_ylabels = [ax.get_ylabel() for ax in fig_mpl.axes]
        png_path = plots_dir / f"{plot_name}.png"
        fig_mpl.savefig(png_path, bbox_inches="tight", dpi=120)
        plt.close(fig_mpl)
        named_trace_x = {
            t.name: [int(x) if isinstance(x, (int, np.integer)) else x for x in t.x]
            for t in fig_plotly.data
            if getattr(t, "name", None) and getattr(t, "x", None) is not None
        }
        plot_inventory[plot_name] = {
            "plotly_traces": len(fig_plotly.data),
            "plotly_title": fig_plotly.layout.title.text if fig_plotly.layout.title else None,
            "plotly_named_trace_x": named_trace_x,
            "matplotlib_axes_count": mpl_axes_count,
            "matplotlib_ylabels": mpl_ylabels,
            "matplotlib_png": str(png_path.relative_to(REPO_ROOT)),
        }

    with open(out_dir / "plot_inventory.json", "w", encoding="utf-8") as f:
        json.dump(plot_inventory, f, indent=2)

    return {
        "dataset_label": dataset_label,
        "n_observations": int(len(df)),
        "n_variables": len(LCA_SYMPTOM_VARS),
        "timings": timings,
        "plots_available_no_bootstrap": list(model_no_bootstrap.plots_map.keys()),
        "plots_available_bootstrap": list(model_bootstrap.plots_map.keys()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate LCA baseline artifacts in artifacts/lca_baseline/."
    )
    parser.add_argument(
        "--clinical-data",
        type=str,
        default=None,
        help="Optional path to external clinical feather dataset (or set RAPID_LCA_DATA_PATH).",
    )
    parser.add_argument(
        "--skip-clinical",
        action="store_true",
        help="Run only synthetic data scenarios even if a clinical dataset is found.",
    )
    args = parser.parse_args()

    base_dir = REPO_ROOT / "artifacts" / "lca_baseline"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment & Git info (using relative paths where possible)
    pkgs = [
        "isaric",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "stepmix",
        "statsmodels",
        "lifelines",
        "matplotlib",
        "seaborn",
        "plotly",
        "reportlab",
        "pytest",
        "ipykernel",
        "nbformat",
        "nbclient",
        "pyarrow",
    ]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib.metadata.version(p)
        except Exception:
            versions[p] = "not_installed"

    isaric_path = Path(isa.__file__).resolve()
    try:
        isaric_rel = str(isaric_path.relative_to(REPO_ROOT))
    except ValueError:
        isaric_rel = str(isaric_path)

    env_info = {
        "python_version": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "git_branch": _git_out(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit_short": _git_out(["rev-parse", "--short", "HEAD"]),
        "git_commit_full": _git_out(["rev-parse", "HEAD"]),
        "git_commit_subject": _git_out(["log", "-1", "--pretty=%s"]),
        "rapid_version": isa.__version__,
        "isaric_file_relative": isaric_rel,
        "isaric_is_local_repo": isaric_path == (REPO_ROOT / "src" / "isaric" / "__init__.py").resolve(),
        "library_versions": versions,
    }
    with open(base_dir / "environment_info.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)

    # 2. Run Synthetic Dataset Scenarios
    df_synth = generate_synthetic_lca_data(n_samples=150, seed=42, symptom_vars=LCA_SYMPTOM_VARS)
    synth_summary = run_dataset_scenarios(
        df=df_synth,
        dataset_label="synthetic_generator (n=150, seed=42)",
        out_dir=base_dir / "synthetic",
    )

    # 3. Run Optional Clinical Dataset Scenarios (if available and not skipped)
    clinical_summary = None
    if not args.skip_clinical:
        feather_path = resolve_optional_clinical_path(args.clinical_data)
        if feather_path is not None:
            df_clin = pd.read_feather(feather_path)
            clinical_summary = run_dataset_scenarios(
                df=df_clin,
                dataset_label=f"clinical_feather ({feather_path.name}, n={len(df_clin)})",
                out_dir=base_dir / "clinical_sample150",
            )

    timings_payload = {
        "synthetic": synth_summary,
        "clinical_sample150": clinical_summary,
    }
    with open(base_dir / "execution_timings.json", "w", encoding="utf-8") as f:
        json.dump(timings_payload, f, indent=2)

    print("Baseline artifacts generated successfully in:", base_dir.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
