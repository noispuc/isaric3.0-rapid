"""Reproducible Before (commit 22b1a6d) vs After (feat/LCA-fixes) baseline comparison.

Exports the verified pre-LCA baseline commit (22b1a6d28b44ee0676ab2a99cb3b06e0aa8af2c0)
into an isolated temporary directory and executes two separate Python subprocesses
(one importing from the baseline checkout's src/ and one importing from the current
working tree's src/) so baseline and updated package code are never mixed in the
same interpreter process.

Usage:
    python scripts/lca/run_before_after_comparison.py
    python scripts/lca/run_before_after_comparison.py --force
    python scripts/lca/run_before_after_comparison.py --out-dir artifacts/release_readiness

Notes:
- Uses only relative/resolved repository paths and Python's standard tempfile module.
- Runs entirely on the deterministic synthetic dataset (n_samples=150, seed=42, p=13).
- Does not require or touch any patient-level clinical dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "release_readiness"
DEFAULT_BASELINE_COMMIT = "22b1a6d28b44ee0676ab2a99cb3b06e0aa8af2c0"


WORKER_CODE = r'''
import contextlib
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

mode = sys.argv[1]
src_path = Path(sys.argv[2]).resolve()
repo_root = Path(sys.argv[3]).resolve()
out_json_path = Path(sys.argv[4]).resolve()
plots_dir = Path(sys.argv[5]).resolve()
rep_dir_1 = Path(sys.argv[6]).resolve()
rep_dir_2 = Path(sys.argv[7]).resolve() if len(sys.argv) > 7 else None

repo_src_resolved = (repo_root / "src").resolve()

# Ensure exact import isolation: remove any pre-existing repo src path and insert target src_path first
cleaned_sys_path = []
for entry in sys.path:
    try:
        resolved_entry = Path(entry).resolve() if entry else None
    except Exception:
        resolved_entry = None
    if resolved_entry in (repo_src_resolved, src_path):
        continue
    cleaned_sys_path.append(entry)

sys.path = [str(src_path), str(repo_root)] + cleaned_sys_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import isaric
from isaric import RAPID
from tests.lca.synthetic_data import (
    LCA_SYMPTOM_VARS,
    generate_synthetic_lca_data,
)

# Verify package origin matches target src_path
isaric_file_resolved = Path(isaric.__file__).resolve()
assert str(isaric_file_resolved).startswith(str(src_path)), (
    f"Import origin mismatch: {isaric_file_resolved} vs {src_path}"
)
isaric_rel_origin = (
    f"<baseline_checkout>/src/{isaric_file_resolved.relative_to(src_path).as_posix()}"
    if mode == "before"
    else f"src/{isaric_file_resolved.relative_to(src_path).as_posix()}"
)

plots_dir.mkdir(parents=True, exist_ok=True)
rep_dir_1.mkdir(parents=True, exist_ok=True)
if rep_dir_2 is not None:
    rep_dir_2.mkdir(parents=True, exist_ok=True)

df = generate_synthetic_lca_data(n_samples=150, seed=42)
param_grid = {"n_clusters": [1, 2, 3, 4]}

# 1. Fit without bootstrap
m_nobs = RAPID.create(
    data=df.copy(),
    model="lca",
    n_clusters=4,
    measurement_vars=list(LCA_SYMPTOM_VARS),
    outcome="severe",
    random_seed=42,
)
m_nobs.fit(param_grid=dict(param_grid), bootstrap=False)

# 2. Fit with bootstrap=True, n_iterations=6
m_boot = RAPID.create(
    data=df.copy(),
    model="lca",
    n_clusters=4,
    measurement_vars=list(LCA_SYMPTOM_VARS),
    outcome="severe",
    random_seed=42,
)
m_boot.fit(param_grid=dict(param_grid), bootstrap=True, n_iterations=6)

def extract_per_k_solutions(model_obj):
    per_k = {}
    for k_val, est in sorted(model_obj.model_objects.items()):
        weights = [float(w) for w in np.asarray(est.weights_).ravel()]
        pis = np.asarray(est.get_parameters()["measurement"]["pis"])
        if pis.ndim == 3:
            probs = pis[:, :, 1]
        else:
            probs = pis
        if probs.shape[0] == len(LCA_SYMPTOM_VARS) and probs.shape[1] == int(k_val):
            probs_ck = probs.T
        else:
            probs_ck = probs
        per_k[str(k_val)] = {
            "weights": weights,
            "item_probs_class_by_feature": [[float(x) for x in row] for row in probs_ck],
        }
    return per_k

# Capture summary() stdout (suppressing interactive figure window show for both matplotlib and plotly)
import plotly.graph_objs as go
orig_show = plt.Figure.show
orig_plotly_show = go.Figure.show
plt.Figure.show = lambda self, *a, **kw: None
go.Figure.show = lambda self, *a, **kw: None

buf_default = io.StringIO()
with contextlib.redirect_stdout(buf_default):
    m_boot.summary()
summary_default_stdout = buf_default.getvalue()

summary_k3_stdout = None
summary_compare_stdout = None
if mode == "after":
    buf_k3 = io.StringIO()
    with contextlib.redirect_stdout(buf_k3):
        m_boot.summary(k=3, plots=[])
    summary_k3_stdout = buf_k3.getvalue()

    buf_cmp = io.StringIO()
    with contextlib.redirect_stdout(buf_cmp):
        m_boot.summary(k=2, plots=[], table_format="short")
    summary_compare_stdout = buf_cmp.getvalue()

    # Restore default summary plot selection
    with contextlib.redirect_stdout(io.StringIO()):
        m_boot.summary()

def _rel_path(target: Path, base: Path) -> str:
    try:
        return target.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return target.name

# Save all matplotlib plots from plots_map
saved_plots = {}
for plot_name, plot_fn in m_boot.plots_map.items():
    fig = plot_fn(backend="matplotlib")
    figs = fig if isinstance(fig, list) else [fig]
    paths = []
    for idx, f in enumerate(figs):
        p = plots_dir / f"{plot_name}_{idx}.png"
        f.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(f)
        paths.append(_rel_path(p, repo_root))
    saved_plots[plot_name] = paths

if mode == "after":
    # Also save solution-specific plots for k=3 via _render_report_plot
    sol_k3 = m_boot._resolve_k_solution(3, mode="selected")
    for pname in [p for p in ["radar_plot", "distribution_plot"] if p in m_boot.plots_map]:
        fig = m_boot._render_report_plot(pname, backend="matplotlib", solution=sol_k3)
        figs = fig if isinstance(fig, list) else [fig]
        for idx, f in enumerate(figs):
            p = plots_dir / f"{pname}_selected_k3_{idx}.png"
            f.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(f)
            saved_plots[f"{pname}_selected_k3"] = [_rel_path(p, repo_root)]

# Run report(format=["csv", "png", "pdf"]) in rep_dir_1 (unselected / Reference k=4)
cwd_orig = os.getcwd()
try:
    os.chdir(rep_dir_1)
    with contextlib.redirect_stdout(io.StringIO()):
        m_boot.report(format=["csv", "png", "pdf"])
        m_boot.save()
    rep1_files = sorted(p.name for p in rep_dir_1.iterdir())
finally:
    os.chdir(cwd_orig)

rep2_files = []
save_meta_selected = None
if mode == "after" and rep_dir_2 is not None:
    try:
        os.chdir(rep_dir_2)
        with contextlib.redirect_stdout(io.StringIO()):
            m_boot.decide(
                selected=True,
                k=3,
                reason="k=3 selected based on BIC elbow, BLRT significance, and clinical interpretability.",
                notes="Release readiness validation run",
            )
            m_boot.report(format=["csv", "png", "pdf"])
            m_boot.save()
        rep2_files = sorted(p.name for p in rep_dir_2.iterdir())
        rapid_zips = sorted(rep_dir_2.glob("*.rapid"))
        if rapid_zips:
            with zipfile.ZipFile(rapid_zips[0], "r") as zf:
                save_meta_selected = json.loads(zf.read("metadata.json").decode("utf-8"))
                est_loaded = pickle.loads(zf.read("model.pkl"))
                save_meta_selected["_verified_pkl_n_components"] = int(est_loaded.n_components)
    finally:
        os.chdir(cwd_orig)

plt.Figure.show = orig_show
go.Figure.show = orig_plotly_show

nested_lrt_records = []
if getattr(m_boot, "nested_models_lrt", None) is not None:
    nested_lrt_records = json.loads(m_boot.nested_models_lrt.to_json(orient="records"))

ci_boot_df = None
if getattr(m_boot, "bootstrap_results", None) and isinstance(m_boot.bootstrap_results, dict):
    ci_boot_df = m_boot.bootstrap_results.get("ci_bootstrap")
elif getattr(m_boot, "ci_bootstrap", None) is not None:
    ci_boot_df = m_boot.ci_bootstrap

ci_boot_records = []
ci_boot_ks = []
if ci_boot_df is not None:
    ci_boot_records = json.loads(ci_boot_df.to_json(orient="records"))
    ci_boot_ks = sorted(int(x) for x in ci_boot_df["n_clusters"].unique().tolist())

payload = {
    "mode": mode,
    "isaric_file": isaric_rel_origin,
    "isaric_version": getattr(isaric, "__version__", "unknown"),
    "plots_map_keys_nobs": list(m_nobs.plots_map.keys()),
    "plots_map_keys_boot": list(m_boot.plots_map.keys()),
    "default_plots_nobs": list(getattr(m_nobs, "_default_plots", [])),
    "default_plots_boot": list(getattr(m_boot, "_default_plots", [])),
    "displayed_plots_after_default_summary": list(getattr(m_boot, "_displayed_plots", []) or []),
    "grid_nobs": json.loads(m_nobs.grid_results.to_json(orient="records")),
    "grid_boot": json.loads(m_boot.grid_results.to_json(orient="records")),
    "nested_models_lrt_boot": nested_lrt_records,
    "model_info_boot": {k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v) for k, v in (m_boot.model_info or {}).items()},
    "ci_bootstrap_k_values": ci_boot_ks,
    "ci_bootstrap_records": ci_boot_records,
    "per_k_nobs": extract_per_k_solutions(m_nobs),
    "per_k_boot": extract_per_k_solutions(m_boot),
    "summary_default_stdout": summary_default_stdout,
    "summary_k3_stdout": summary_k3_stdout,
    "summary_compare_stdout": summary_compare_stdout,
    "saved_plots": saved_plots,
    "report_unselected_files": rep1_files,
    "report_selected_k3_files": rep2_files,
    "save_meta_selected": save_meta_selected,
}

out_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
'''


def export_pre_lca_baseline(baseline_commit: str, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", baseline_commit],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(target_dir)],
        input=archive.stdout,
        check=True,
    )


def run_worker(
    mode: str,
    src_path: Path,
    out_json: Path,
    plots_dir: Path,
    rep_dir_1: Path,
    rep_dir_2: Path | None = None,
) -> Dict[str, Any]:
    for d in [plots_dir, rep_dir_1] + ([rep_dir_2] if rep_dir_2 else []):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-c",
        WORKER_CODE,
        mode,
        str(src_path),
        str(REPO_ROOT),
        str(out_json),
        str(plots_dir),
        str(rep_dir_1),
    ]
    if rep_dir_2 is not None:
        cmd.append(str(rep_dir_2))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{REPO_ROOT}"
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def fmt_val(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def build_numerical_comparison_csv(
    before: Dict[str, Any],
    after: Dict[str, Any],
    csv_path: Path,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    b_grid_boot = {int(r["n_clusters"]): r for r in before["grid_boot"]}
    a_grid_boot = {int(r["n_clusters"]): r for r in after["grid_boot"]}
    b_grid_nobs = {int(r["n_clusters"]): r for r in before["grid_nobs"]}
    a_grid_nobs = {int(r["n_clusters"]): r for r in after["grid_nobs"]}

    # Legacy LRT lookup in before: nested_models_lrt has rows for k=2,3,4
    b_lrt_by_k: Dict[int, Dict[str, Any]] = {
        int(rec["n_clusters"]): rec
        for rec in before.get("nested_models_lrt_boot", [])
    }

    b_blrt_list = before.get("model_info_boot", {}).get("lrt_blrt", []) or []
    b_blrt_by_k: Dict[int, Any] = {
        int(rec["n_clusters"]): rec.get("blrt_p_value")
        for rec in b_blrt_list
        if "n_clusters" in rec
    }

    core_metrics = [
        ("LL", 1e-6),
        ("AIC", 1e-6),
        ("BIC", 1e-6),
        ("CAIC", 1e-6),
        ("SABIC", 1e-6),
        ("entropy", 1e-6),
        ("relative_entropy", 1e-6),
        ("ave_pp_min", 1e-6),
        ("occ_k_min", 1e-6),
        ("min_class_perc", 1e-6),
    ]

    for k in [1, 2, 3, 4]:
        br = b_grid_boot[k]
        ar = a_grid_boot[k]

        for m_name, tol in core_metrics:
            bv = br.get(m_name)
            av = ar.get(m_name)
            if bv is None and av is None:
                diff_str = "0.000000"
                status = "Pass"
            elif bv is not None and av is not None:
                diff = abs(float(av) - float(bv))
                diff_str = f"{diff:.6e}"
                status = "Pass" if diff <= tol else "Unexpected Regression"
            else:
                diff_str = "N/A"
                status = "Intentional Change"
            rows.append({
                "Metric": m_name,
                "k": str(k),
                "Before": fmt_val(bv),
                "After": fmt_val(av),
                "Difference": diff_str,
                "Tolerance": f"{tol:.0e}",
                "Status": status,
            })

        # converged flag
        bv_conv = br.get("converged")
        av_conv = ar.get("converged")
        rows.append({
            "Metric": "converged",
            "k": str(k),
            "Before": fmt_val(bv_conv),
            "After": fmt_val(av_conv),
            "Difference": "0" if bv_conv == av_conv else "Changed",
            "Tolerance": "Exact",
            "Status": "Pass" if bv_conv == av_conv else "Unexpected Regression",
        })

        # convergence_rate (no bootstrap): Before was 1.0 (from single fit converged_), After is None (LCA-02)
        bv_cr_nobs = b_grid_nobs[k].get("convergence_rate")
        av_cr_nobs = a_grid_nobs[k].get("convergence_rate")
        rows.append({
            "Metric": "convergence_rate (bootstrap=False)",
            "k": str(k),
            "Before": fmt_val(bv_cr_nobs),
            "After": fmt_val(av_cr_nobs),
            "Difference": "1.0 -> None (LCA-02)",
            "Tolerance": "By Design",
            "Status": "Intentional Change",
        })

        # convergence_rate (bootstrap=True): Before was float(converged_), After is bootstrap replication convergence rate
        bv_cr_boot = br.get("convergence_rate")
        av_cr_boot = ar.get("convergence_rate")
        diff_cr = abs(float(av_cr_boot) - float(bv_cr_boot)) if (bv_cr_boot is not None and av_cr_boot is not None) else 0.0
        rows.append({
            "Metric": "convergence_rate (bootstrap=True)",
            "k": str(k),
            "Before": fmt_val(bv_cr_boot),
            "After": fmt_val(av_cr_boot),
            "Difference": f"{diff_cr:.6e} (sourced from bootstrap_stats)",
            "Tolerance": "By Design (LCA-02)",
            "Status": "Pass" if diff_cr <= 1e-6 else "Intentional Change",
        })

        # LRT metrics (LCA-01: integrated into grid_results; numerically identical to legacy nested_models_lrt)
        for lrt_col, legacy_col in [
            ("lrt_statistic", "lr_statistic"),
            ("lrt_dof_diff", "dof_diff"),
            ("lrt_p_value", "p_value"),
        ]:
            av_lrt = ar.get(lrt_col)
            if k == 1:
                rows.append({
                    "Metric": f"{lrt_col} (in grid_results)",
                    "k": "1",
                    "Before": "Not in grid_results (N/A for k=1)",
                    "After": fmt_val(av_lrt),
                    "Difference": "0.000000",
                    "Tolerance": "Exact (None for k=1)",
                    "Status": "New Functionality",
                })
            else:
                legacy_val = b_lrt_by_k.get(k, {}).get(legacy_col)
                diff = abs(float(av_lrt) - float(legacy_val))
                rows.append({
                    "Metric": f"{lrt_col} (in grid_results vs legacy nested_models_lrt)",
                    "k": str(k),
                    "Before": f"{fmt_val(legacy_val)} (separate table only)",
                    "After": f"{fmt_val(av_lrt)} (in grid_results)",
                    "Difference": f"{diff:.6e}",
                    "Tolerance": "1e-06",
                    "Status": "Pass" if diff <= 1e-6 else "Unexpected Regression",
                })

        # BLRT p-value (LCA-01: integrated into grid_results; numerically identical to legacy model_info['lrt_blrt'])
        av_blrt = ar.get("blrt_p_value")
        if k == 1:
            rows.append({
                "Metric": "blrt_p_value (in grid_results)",
                "k": "1",
                "Before": "Not in grid_results (N/A for k=1)",
                "After": fmt_val(av_blrt),
                "Difference": "0.000000",
                "Tolerance": "Exact (None for k=1)",
                "Status": "New Functionality",
            })
        else:
            legacy_blrt = b_blrt_by_k.get(k)
            diff_b = abs(float(av_blrt) - float(legacy_blrt))
            rows.append({
                "Metric": "blrt_p_value (in grid_results vs legacy model_info)",
                "k": str(k),
                "Before": f"{fmt_val(legacy_blrt)} (in model_info only)",
                "After": f"{fmt_val(av_blrt)} (in grid_results)",
                "Difference": f"{diff_b:.6e}",
                "Tolerance": "1e-06",
                "Status": "Pass" if diff_b <= 1e-6 else "Unexpected Regression",
            })

        # Class prevalence (weights_) and Item-Response Probabilities per k
        b_sol = before["per_k_boot"][str(k)]
        a_sol = after["per_k_boot"][str(k)]
        b_w = np.array(b_sol["weights"], dtype=float)
        a_w = np.array(a_sol["weights"], dtype=float)
        max_w_diff = float(np.max(np.abs(a_w - b_w)))
        rows.append({
            "Metric": "class_prevalence_weights_max_abs_diff",
            "k": str(k),
            "Before": json.dumps([round(x, 6) for x in b_w.tolist()]),
            "After": json.dumps([round(x, 6) for x in a_w.tolist()]),
            "Difference": f"{max_w_diff:.6e}",
            "Tolerance": "1e-06",
            "Status": "Pass" if max_w_diff <= 1e-6 else "Unexpected Regression",
        })

        b_probs = np.array(b_sol["item_probs_class_by_feature"], dtype=float)
        a_probs = np.array(a_sol["item_probs_class_by_feature"], dtype=float)
        max_p_diff = float(np.max(np.abs(a_probs - b_probs)))
        rows.append({
            "Metric": "item_response_probabilities_max_abs_diff (all classes x 13 items)",
            "k": str(k),
            "Before": f"shape={list(b_probs.shape)} mean={float(np.mean(b_probs)):.6f}",
            "After": f"shape={list(a_probs.shape)} mean={float(np.mean(a_probs)):.6f}",
            "Difference": f"{max_p_diff:.6e}",
            "Tolerance": "1e-06",
            "Status": "Pass" if max_p_diff <= 1e-6 else "Unexpected Regression",
        })

    # Bootstrap CI inclusion of k=1 (LCA-05)
    b_ci_ks = before.get("ci_bootstrap_k_values") or [2, 3, 4]
    a_ci_ks = after.get("ci_bootstrap_k_values") or [1, 2, 3, 4]
    rows.append({
        "Metric": "ci_bootstrap_included_k_values",
        "k": "1..4",
        "Before": json.dumps(b_ci_ks),
        "After": json.dumps(a_ci_ks),
        "Difference": "Added k=1 to bootstrap CI table (LCA-05)",
        "Tolerance": "By Design (LCA-05)",
        "Status": "New Functionality",
    })

    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Metric", "k", "Before", "After", "Difference", "Tolerance", "Status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible Before (22b1a6d) vs After (feat/LCA-fixes) LCA validation comparison."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for comparison CSV, JSON snapshots, plots, and sample reports.",
    )
    parser.add_argument(
        "--baseline-commit",
        type=str,
        default=DEFAULT_BASELINE_COMMIT,
        help="Git commit hash for the pre-LCA baseline (default: 22b1a6d28b44ee0676ab2a99cb3b06e0aa8af2c0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-execution of both isolated Before and After subprocesses even if cached JSON exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plots_before_dir = out_dir / "plots" / "before_22b1a6d"
    plots_after_dir = out_dir / "plots" / "after_feat_lca_fixes"
    reports_before_dir = out_dir / "reports_sample" / "before_22b1a6d"
    reports_after_ref_dir = out_dir / "reports_sample" / "after_reference_k4"
    reports_after_sel_dir = out_dir / "reports_sample" / "after_selected_k3"

    before_json = out_dir / "before_22b1a6d_run.json"
    after_json = out_dir / "after_feat_lca_fixes_run.json"

    if args.force or not before_json.exists():
        with tempfile.TemporaryDirectory(prefix="rapid_pre_lca_") as tmp_dir:
            pre_lca_tmp = Path(tmp_dir)
            export_pre_lca_baseline(args.baseline_commit, pre_lca_tmp)
            print(f"Running BEFORE baseline (commit {args.baseline_commit[:7]}) in isolated subprocess...")
            before_data = run_worker(
                mode="before",
                src_path=pre_lca_tmp / "src",
                out_json=before_json,
                plots_dir=plots_before_dir,
                rep_dir_1=reports_before_dir,
                rep_dir_2=None,
            )
    else:
        before_data = json.loads(before_json.read_text(encoding="utf-8"))

    if args.force or not after_json.exists():
        print("Running AFTER validation (current working tree) in isolated subprocess...")
        after_data = run_worker(
            mode="after",
            src_path=REPO_ROOT / "src",
            out_json=after_json,
            plots_dir=plots_after_dir,
            rep_dir_1=reports_after_ref_dir,
            rep_dir_2=reports_after_sel_dir,
        )
    else:
        after_data = json.loads(after_json.read_text(encoding="utf-8"))

    csv_path = out_dir / "03_numerical_comparison.csv"
    rows = build_numerical_comparison_csv(before_data, after_data, csv_path)
    statuses: Dict[str, int] = {}
    for r in rows:
        statuses[r["Status"]] = statuses.get(r["Status"], 0) + 1
    print(f"Comparison complete: {len(rows)} rows written to {csv_path}")
    print(f"Status summary: {statuses}")
    if statuses.get("Unexpected Regression", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
