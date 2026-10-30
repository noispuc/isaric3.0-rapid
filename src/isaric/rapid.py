"""
RAPID: Reusable Analytical Pipelines for Infectious Diseases.

This module provides the abstract base class that defines the contract
for all analytical pipelines in the ISARIC RAPID framework.

Lifecycle:
  1. create()     - Configure and instantiate the pipeline
  2. fit()        - Train the model and compute evaluation metrics

Available after fit():
  - summary()    - Display results and visualizations
  - save()       - Persist the trained model
  - validation() - Validate with independent external data
  - report()     - Generate publication-ready report (classmethod)
  - decide()     - Document model selection decision

State control:
  created → fitted

  summary(), save(), validation(), decide() require fitted state.
  report() is a classmethod — validates all models are fitted.
  fit() can be called again to re-fit (overwrites previous results).

Contract Note:
    Concrete methods access standardized attributes from subclasses.
    Each subclass MUST provide:
    - self._model (configured model from create)
    - self.X, self.y (data matrices)
    - self.fitted_model (trained model)
    - self.result_df (results DataFrame)
    - self.metrics (dict with performance metrics)
    - self.plots_map (dict: plot_name → plot_function)
    - self.model_type (str)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pandas import DataFrame


class RAPID(ABC):
    """
    Abstract base class for all RAPID analytical pipelines.

    This class defines the contract and provides state control.
    Subclasses implement create() (abstract) and inherit concrete
    methods (fit, summary, save, validation, report, decide).
    """

    # ======================================================================
    # STATE CONSTANTS
    # ======================================================================

    _STATE_CREATED = "created"
    _STATE_FITTED  = "fitted"

    # Model families — used to validate summary(models=[...]) comparisons
    _FAMILY_MAP = {
        "logistic":       "inferential",
        "glm":            "inferential",
        "survival":       "survival",
        "lca":            "clustering",
        "kmeans":         "clustering",
        "decision_tree":  "tree_based",
        "random_forest":  "tree_based",
        "xgboost":        "tree_based",
        "lightgbm":       "tree_based",
        "catboost":       "tree_based",
        "lasso":          "predictive",
        "ridge":          "predictive",
        "elastic_net":    "predictive",
        "svm":            "predictive",
        "logistic_l2":    "predictive",
        "descriptive":    "descriptive",
    }

    # Classification models — only these activate balancing/threshold/recalibration
    _CLASSIFICATION_MODELS = {
        'logistic', 'glm', 'lasso', 'ridge', 'elastic_net',
        'svm', 'logistic_l2', 'decision_tree', 'random_forest',
        'xgboost', 'lightgbm', 'catboost',
    }

    # ======================================================================
    # CONSTRUCTOR (INHERITED BY SUBCLASSES)
    # ======================================================================

    def __init__(self):
        """
        Initialize pipeline with created state.

        Attributes set here are inherited by all subclasses.
        model_id is set by create() after instantiation.
        """
        self._state = self._STATE_CREATED
        self._libraries_used = {}
        self._fit_at = None
        self._random_seed = 42      # default; subclasses podem sobrescrever
        self.model_id = None        # set by create() — auto-generated
        self._validation_log = []   # accumulates each validation() call
        self._decision = None       # set by decide() — documents selection
        self._displayed_plots = []  # set by summary() — used by report()

        # Train/test split (set by fit() when test_size is provided)
        self.X_train      = None
        self.X_test       = None
        self.y_train      = None
        self.y_test       = None
        self.X_train_full = None    # X_train before balancing (for recalibration)
        self.y_train_full = None    # y_train before balancing

        # Predictive model parameters (set by fit())
        self._threshold           = 0.5
        self._resolved_threshold  = 0.5     # actual threshold after "f1"/"youden" resolution
        self._balancing           = None
        self._recalibrate         = False
        self._recalibrate_on      = "train"
        self._recalibration_slope     = None
        self._recalibration_intercept = None

    # ======================================================================
    # STATE CONTROL (INHERITED BY SUBCLASSES)
    # ======================================================================

    def _check_state(
        self,
        required_state,
        method_name: str,
    ) -> None:
        """
        Validate that the pipeline is in the correct state.

        Args:
            required_state: A single state string or a list of accepted
                state strings.
            method_name: Name of the method being called.

        Raises:
            ValueError: If the pipeline is not in the required state.
        """
        if isinstance(required_state, list):
            if self._state not in required_state:
                accepted = "', '".join(required_state)
                raise ValueError(
                    f"Cannot call {method_name}() in state '{self._state}'. "
                    f"Required state: one of '{accepted}'."
                )
        else:
            if self._state != required_state:
                raise ValueError(
                    f"Cannot call {method_name}() in state '{self._state}'. "
                    f"Required state: '{required_state}'."
                )

    def _transition_to(self, new_state: str) -> None:
        """
        Transition the pipeline to a new state.

        Args:
            new_state: The new state to transition to.
        """
        self._state = new_state


    # ======================================================================
    # REGISTER LIBRARIES (INHERITED BY SUBCLASSES)
    # ======================================================================

    def _register_library(self, library_name: str) -> None:
        """
        Register a library used during the pipeline execution.
        
        Args:
            library_name: Name of the library (e.g., "statsmodels", "scikit-learn").
        """
        import importlib.metadata
        
        try:
            version = importlib.metadata.version(library_name)
            self._libraries_used[library_name] = version
        except importlib.metadata.PackageNotFoundError:
            self._libraries_used[library_name] = "not_installed" 

    @staticmethod
    def _apply_balancing(
        X,
        y,
        method: str,
        random_state: int = 42,
    ):
        """
        Apply class balancing to training data.

        Args:
            X: Training predictor matrix.
            y: Training outcome vector.
            method: "undersample" (random undersampling of majority class)
                or "smote" (synthetic minority oversampling).
            random_state: Seed for reproducibility.

        Returns:
            Tuple (X_resampled, y_resampled).

        Raises:
            ImportError: If imbalanced-learn is not installed.
            ValueError: If method is invalid.
        """
        try:
            if method == "undersample":
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=random_state)
            elif method == "smote":
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=random_state)
            else:
                raise ValueError(
                    f"balancing must be 'undersample' or 'smote'. "
                    f"Received: '{method}'"
                )
        except ImportError:
            raise ImportError(
                "imbalanced-learn is required for class balancing. "
                "Install with: pip install imbalanced-learn"
            )

        import pandas as pd
        X_res, y_res = sampler.fit_resample(X, y)

        if isinstance(X, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns)
        if isinstance(y, pd.Series):
            y_res = pd.Series(y_res, name=y.name)

        return X_res, y_res

    @staticmethod
    def _apply_recalibration(
        fitted_model,
        X_cal,
        y_cal,
    ) -> dict:
        """
        Apply logistic recalibration to model predictions.

        Fits a logistic regression of y_cal on logit(y_prob_cal).
        This is monotone — does not change AUC or ranking.
        Changes the probability scale and calibration.

        Args:
            fitted_model: Fitted sklearn model with predict_proba().
            X_cal: Calibration predictor matrix.
            y_cal: Calibration outcome vector.

        Returns:
            Dict with calibration_slope and calibration_intercept.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression as _LR

        if not hasattr(fitted_model, 'predict_proba'):
            return {}

        y_prob = fitted_model.predict_proba(X_cal)[:, 1]
        y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
        log_odds = np.log(y_prob / (1 - y_prob))

        cal = _LR(C=1e6, solver='lbfgs', max_iter=500)
        cal.fit(log_odds.reshape(-1, 1), y_cal)

        return {
            'calibration_slope':     float(cal.coef_[0][0]),
            'calibration_intercept': float(cal.intercept_[0]),
        }

    # ======================================================================
    # COMPARE MODELS (INHERITED BY SUBCLASSES)
    # ======================================================================
    def _summary_comparative(
        self,
        models: List["RAPID"],
    ) -> None:
        """
        Display comparative table for models of the same family.

        Called by summary(compare=[...]) with self as the first model.
        Shows common metrics and common variables/features across models.

        Args:
            models: List of RAPID instances — self is always first.
        """
        import pandas as pd

        model_ids = [m.model_id for m in models]
        width = 80

        print("=" * width)
        print("RAPID — MODEL COMPARISON")
        print(f"Reference : {model_ids[0]}")
        print(f"Comparing : {', '.join(model_ids[1:])}")
        print("=" * width)
        print()

        # ── Metrics ───────────────────────────────────────────────────────
        print("-" * width)
        print("METRICS")
        print("-" * width)

        all_metric_keys = []
        for m in models:
            if hasattr(m, 'metrics') and m.metrics:
                for k, v in m.metrics.items():
                    if isinstance(v, (int, float)) and k not in all_metric_keys:
                        all_metric_keys.append(k)

        if all_metric_keys:
            rows = []
            for key in all_metric_keys:
                row = {"Metric": key}
                for m in models:
                    val = (m.metrics or {}).get(key, None)
                    row[m.model_id] = (
                        f"{val:.4f}" if isinstance(val, float)
                        else str(val) if val is not None
                        else "—"
                    )
                rows.append(row)

            metrics_df = pd.DataFrame(rows).set_index("Metric")
            print(metrics_df.to_string())
            print()

        # ── Common variables / coefficients ───────────────────────────────
        common_vars = None
        var_col_used = None

        for m in models:
            if hasattr(m, 'result_df') and m.result_df is not None:
                var_col = next(
                    (c for c in ['Variable', 'Feature', 'feature']
                     if c in m.result_df.columns),
                    None
                )
                if var_col:
                    var_col_used = var_col
                    vars_m = set(m.result_df[var_col].tolist())
                    common_vars = (
                        vars_m if common_vars is None
                        else common_vars & vars_m
                    )

        if common_vars:
            print("-" * width)
            print(f"COMMON VARIABLES ({len(common_vars)} shared)")
            print("-" * width)

            coef_candidates = [
                'Coefficient', 'OddsRatio', 'HazardRatio',
                'Importance', 'Effect',
            ]

            rows = []
            for var in sorted(common_vars):
                row = {"Variable": var}
                for m in models:
                    if hasattr(m, 'result_df') and m.result_df is not None:
                        vc = next(
                            (c for c in ['Variable', 'Feature', 'feature']
                             if c in m.result_df.columns),
                            None
                        )
                        cc = next(
                            (c for c in coef_candidates
                             if c in m.result_df.columns),
                            None
                        )
                        if vc and cc:
                            match = m.result_df[m.result_df[vc] == var]
                            val = match.iloc[0][cc] if not match.empty else None
                            row[m.model_id] = (
                                f"{val:.4f}" if isinstance(val, float)
                                else str(val) if val is not None
                                else "—"
                            )
                        else:
                            row[m.model_id] = "—"
                rows.append(row)

            if rows:
                coef_df = pd.DataFrame(rows).set_index("Variable")
                print(coef_df.to_string())
                print()

        print("=" * width)
        print(
            "ℹ️  Call model.summary() individually for full details, "
            "plots and grid results."
        )
        print("=" * width)

    # ======================================================================
    # STATIC METHOD: PDF REPORT GENERATION (INHERITED BY SUBCLASSES)
    # ======================================================================
    @staticmethod
    def _generate_pdf_report(
        models: List["RAPID"],
        pdf_filename: str,
    ) -> None:
        """
        Generate consolidated PDF report using ReportLab.

        Called internally by report(). Not intended for direct use.

        Args:
            models: List of fitted RAPID instances.
            pdf_filename: Output PDF file path.

        Raises:
            ImportError: If reportlab is not installed.
        """
        try:
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, PageBreak, HRFlowable, Image,
            )
            from reportlab.lib.styles import (
                getSampleStyleSheet, ParagraphStyle,
            )
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.enums import TA_CENTER
        except ImportError:
            raise ImportError(
                "ReportLab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

        import os
        import tempfile
        import isaric
        from datetime import datetime

        PAGE_W, PAGE_H = A4

        # ── Colours ───────────────────────────────────────────────────────
        BLUE      = colors.HexColor('#003366')
        LIGHT     = colors.HexColor('#e8f0f8')
        GRAY      = colors.HexColor('#cccccc')
        DARK      = colors.HexColor('#222222')

        # ── Styles ────────────────────────────────────────────────────────
        SS = getSampleStyleSheet()

        s_cover_title = ParagraphStyle(
            'CoverTitle', parent=SS['Title'],
            fontSize=22, textColor=BLUE,
            spaceAfter=12, fontName='Helvetica-Bold',
            alignment=TA_CENTER,
        )
        s_cover_sub = ParagraphStyle(
            'CoverSub', parent=SS['Normal'],
            fontSize=11, textColor=DARK,
            spaceAfter=6, alignment=TA_CENTER,
        )
        s_h1 = ParagraphStyle(
            'H1', parent=SS['Heading1'],
            fontSize=14, textColor=BLUE,
            spaceBefore=16, spaceAfter=6,
            fontName='Helvetica-Bold',
        )
        s_h2 = ParagraphStyle(
            'H2', parent=SS['Heading2'],
            fontSize=11, textColor=BLUE,
            spaceBefore=10, spaceAfter=4,
            fontName='Helvetica-Bold',
        )
        s_body = ParagraphStyle(
            'Body', parent=SS['Normal'],
            fontSize=9, leading=13, textColor=DARK,
        )
        s_note = ParagraphStyle(
            'Note', parent=SS['Normal'],
            fontSize=8, textColor=colors.HexColor('#666666'),
            fontName='Helvetica-Oblique',
        )

        # ── Table style helper ────────────────────────────────────────────
        def _tbl_style():
            return TableStyle([
                ('BACKGROUND',    (0, 0), (-1, 0),  BLUE),
                ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
                ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0, 0), (-1, -1), 8),
                ('ROWBACKGROUNDS',(0, 1), (-1, -1),  [colors.white, LIGHT]),
                ('GRID',          (0, 0), (-1, -1),  0.25, GRAY),
                ('TOPPADDING',    (0, 0), (-1, -1),  3),
                ('BOTTOMPADDING', (0, 0), (-1, -1),  3),
            ])

        # ── Header / footer ───────────────────────────────────────────────
        def _hf(canvas, doc):
            canvas.saveState()
            canvas.setStrokeColor(BLUE)
            canvas.setLineWidth(1.5)
            canvas.line(inch, PAGE_H - 0.6*inch, PAGE_W - inch, PAGE_H - 0.6*inch)
            canvas.setFont('Helvetica-Bold', 8)
            canvas.setFillColor(BLUE)
            canvas.drawString(inch, PAGE_H - 0.5*inch,
                              "ISARIC CLINICAL EPIDEMIOLOGY PLATFORM")
            canvas.drawRightString(PAGE_W - inch, PAGE_H - 0.5*inch,
                                   f"RAPID v{isaric.__version__}")
            canvas.setStrokeColor(GRAY)
            canvas.setLineWidth(0.5)
            canvas.line(inch, 0.6*inch, PAGE_W - inch, 0.6*inch)
            canvas.setFont('Helvetica', 7)
            canvas.setFillColor(colors.HexColor('#666666'))
            canvas.drawString(inch, 0.4*inch,
                              datetime.now().strftime('%Y-%m-%d'))
            canvas.drawCentredString(
                PAGE_W / 2, 0.4*inch,
                "RAPID — Reusable Analytics Pipeline for Infectious Disease"
            )
            canvas.drawRightString(PAGE_W - inch, 0.4*inch,
                                   f"Page {doc.page}")
            canvas.restoreState()

        # ── Story ─────────────────────────────────────────────────────────
        story = []

        # Cover
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            "ISARIC CLINICAL EPIDEMIOLOGY PLATFORM", s_cover_title
        ))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("RAPID — Analytical Report", s_cover_sub))
        story.append(HRFlowable(
            width="80%", thickness=2, color=BLUE, spaceAfter=12
        ))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            f"Models: {', '.join(m.model_id for m in models)}", s_cover_sub
        ))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}",
            s_cover_sub
        ))
        story.append(Paragraph(
            f"RAPID version: {isaric.__version__}", s_cover_sub
        ))
        story.append(PageBreak())

        with tempfile.TemporaryDirectory() as tmpdir:

            for model in models:

                # ── Model header ──────────────────────────────────────────
                story.append(Paragraph(f"Model: {model.model_id}", s_h1))
                story.append(Paragraph(
                    f"Type: {model.model_type} | "
                    f"Family: {RAPID._FAMILY_MAP.get(model.model_type, 'unknown')}",
                    s_body
                ))
                story.append(HRFlowable(
                    width="100%", thickness=0.5, color=GRAY, spaceAfter=6
                ))

                # ── Execution context ─────────────────────────────────────
                story.append(Paragraph("Execution Context", s_h2))
                ctx = [["Parameter", "Value"]]
                ctx.append(["model_id",    str(model.model_id)])
                ctx.append(["fit_at",      str(getattr(model, '_fit_at', 'Unknown'))])
                ctx.append(["random_seed", str(getattr(model, '_random_seed', '—'))])
                if hasattr(model, 'model_info') and model.model_info:
                    for k in ['framework', 'method', 'N_observations',
                               'N_events', 'N_features']:
                        if k in model.model_info:
                            ctx.append([k, str(model.model_info[k])])
                t = Table(ctx, colWidths=[2.5*inch, 3.5*inch])
                t.setStyle(_tbl_style())
                story.append(t)
                story.append(Spacer(1, 0.15*inch))

                # ── Fit metrics ───────────────────────────────────────────
                if hasattr(model, 'metrics') and model.metrics:
                    story.append(Paragraph("Fit Metrics", s_h2))
                    met = [["Metric", "Value"]]
                    for k, v in model.metrics.items():
                        if k == 'confusion_matrix':
                            continue
                        met.append([
                            k,
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                        ])
                    t = Table(met, colWidths=[3*inch, 3*inch])
                    t.setStyle(_tbl_style())
                    story.append(t)
                    story.append(Spacer(1, 0.15*inch))

                # ── Parameter table ───────────────────────────────────────
                if hasattr(model, 'result_df') and model.result_df is not None:
                    story.append(Paragraph("Parameter Table", s_h2))
                    df = model.result_df.reset_index(drop=True)
                    rows = [list(df.columns)]
                    for _, row in df.iterrows():
                        rows.append([
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                            for v in row
                        ])
                    col_w = (PAGE_W - 2*inch) / len(rows[0])
                    t = Table(rows, colWidths=[col_w] * len(rows[0]))
                    t.setStyle(_tbl_style())
                    story.append(t)
                    story.append(Spacer(1, 0.15*inch))

                # ── Grid results (LCA / clustering) ──────────────────────────
                if hasattr(model, 'grid_results') and model.grid_results is not None:
                    story.append(Paragraph("Class Enumeration Results", s_h2))
                    gdf = model.grid_results.reset_index(drop=True)
                    # Select most relevant columns for the report
                    show_cols = [c for c in [
                        'n_clusters', 'LL', 'AIC', 'BIC', 'CAIC', 'SABIC',
                        'entropy', 'relative_entropy', 'convergence_rate',
                        'ave_pp_min', 'min_class_perc',
                    ] if c in gdf.columns]
                    gdf_show = gdf[show_cols] if show_cols else gdf
                    rows = [list(gdf_show.columns)]
                    for _, row in gdf_show.iterrows():
                        rows.append([
                            f"{v:.3f}" if isinstance(v, float) else str(v)
                            for v in row
                        ])
                    col_w = (PAGE_W - 2*inch) / max(len(rows[0]), 1)
                    t = Table(rows, colWidths=[col_w] * len(rows[0]))
                    t.setStyle(_tbl_style())
                    story.append(t)
                    story.append(Spacer(1, 0.15*inch))

                # ── Nested LRT ────────────────────────────────────────────────
                if (hasattr(model, 'nested_models_lrt')
                        and model.nested_models_lrt is not None
                        and not model.nested_models_lrt.empty):
                    story.append(Paragraph("Nested LRT (Chi-square)", s_h2))
                    ndf = model.nested_models_lrt.reset_index(drop=True)
                    rows = [list(ndf.columns)]
                    for _, row in ndf.iterrows():
                        rows.append([
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                            for v in row
                        ])
                    col_w = (PAGE_W - 2*inch) / max(len(rows[0]), 1)
                    t = Table(rows, colWidths=[col_w] * len(rows[0]))
                    t.setStyle(_tbl_style())
                    story.append(t)
                    story.append(Spacer(1, 0.15*inch))

                # ── Class diagnostics (LCA) ───────────────────────────────────
                if hasattr(model, 'DiagClass') and model.DiagClass:
                    story.append(Paragraph("Class Diagnostics", s_h2))

                    # Class sizes/proportions
                    try:
                        weights = model.fitted_model.weights_
                        n_classes = len(weights)
                        class_labels = (
                            model.class_names
                            if hasattr(model, 'class_names')
                                and model.class_names
                                and len(model.class_names) == n_classes
                            else [f"Class_{i+1}" for i in range(n_classes)]
                        )
                        sizes_data = [["Class", "Proportion", "N (est.)"]]
                        n_obs = len(model.X)
                        for lbl, w in zip(class_labels, weights):
                            sizes_data.append([
                                lbl,
                                f"{w:.4f}",
                                f"{int(round(w * n_obs))}",
                            ])
                        t = Table(sizes_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
                        t.setStyle(_tbl_style())
                        story.append(Paragraph("Class Sizes", s_body))
                        story.append(t)
                        story.append(Spacer(1, 0.1*inch))
                    except Exception:
                        pass

                    # Average Posterior Probabilities
                    ave_pp = model.DiagClass.get('ave_pp')
                    if ave_pp is not None and not ave_pp.empty:
                        story.append(Paragraph(
                            "Average Posterior Probabilities (AvePP)", s_body
                        ))
                        rows = [["k"] + list(ave_pp.columns)]
                        for idx, row in ave_pp.iterrows():
                            rows.append(
                                [str(idx)] + [
                                    f"{v:.4f}" if isinstance(v, float) else str(v)
                                    for v in row
                                ]
                            )
                        col_w = (PAGE_W - 2*inch) / max(len(rows[0]), 1)
                        t = Table(rows, colWidths=[col_w] * len(rows[0]))
                        t.setStyle(_tbl_style())
                        story.append(t)
                        story.append(Spacer(1, 0.1*inch))

                    # OCC ratios
                    occ = model.DiagClass.get('occ_ratios')
                    if occ is not None and not occ.empty:
                        story.append(Paragraph(
                            "Odds of Correct Classification (OCC)", s_body
                        ))
                        rows = [["k"] + list(occ.columns)]
                        for idx, row in occ.iterrows():
                            rows.append(
                                [str(idx)] + [
                                    f"{v:.4f}" if isinstance(v, float) else str(v)
                                    for v in row
                                ]
                            )
                        col_w = (PAGE_W - 2*inch) / max(len(rows[0]), 1)
                        t = Table(rows, colWidths=[col_w] * len(rows[0]))
                        t.setStyle(_tbl_style())
                        story.append(t)
                        story.append(Spacer(1, 0.1*inch))

                # ── Outcome crosstab ──────────────────────────────────────────
                if (hasattr(model, 'outcome_crosstab_df')
                        and model.outcome_crosstab_df is not None):
                    outcome_name = getattr(model, 'outcome', 'outcome')
                    story.append(Paragraph(
                        f"Outcome by Class — {outcome_name}", s_h2
                    ))
                    ct = model.outcome_crosstab_df.reset_index()
                    rows = [list(ct.columns)]
                    for _, row in ct.iterrows():
                        rows.append([
                            f"{v:.4f}" if isinstance(v, float) else str(v)
                            for v in row
                        ])
                    col_w = (PAGE_W - 2*inch) / max(len(rows[0]), 1)
                    t = Table(rows, colWidths=[col_w] * len(rows[0]))
                    t.setStyle(_tbl_style())
                    story.append(t)
                    if (hasattr(model, 'outcome_test_result')
                            and model.outcome_test_result):
                        tr = model.outcome_test_result
                        story.append(Paragraph(
                            f"{tr.get('test', '')}: "
                            f"stat={tr.get('statistic', float('nan')):.4f}, "
                            f"p={tr.get('p_value', float('nan')):.4f}",
                            s_note
                        ))
                    story.append(Spacer(1, 0.15*inch))

                # ── Validation log ────────────────────────────────────────────
                if model._validation_log:
                    story.append(Paragraph("Validation Log", s_h2))
                    for i, entry in enumerate(model._validation_log, 1):
                        story.append(Paragraph(
                            f"Validation {i} — {entry.get('timestamp', '')}",
                            s_body
                        ))
                        for k, v in entry.get('results', {}).items():
                            story.append(Paragraph(
                                f"  {k}: {str(v)[:300]}", s_note
                            ))
                    story.append(Spacer(1, 0.1*inch))

                # ── Decision ──────────────────────────────────────────────
                if model._decision:
                    story.append(Paragraph("Decision", s_h2))
                    story.append(Paragraph(
                        f"Selected: {model._decision.get('selected', '—')}",
                        s_body
                    ))
                    if model._decision.get('reason'):
                        story.append(Paragraph(
                            f"Reason: {model._decision['reason']}", s_body
                        ))
                    story.append(Spacer(1, 0.1*inch))

                # ── Figures ───────────────────────────────────────────────
                plots_to_use = (
                    model._displayed_plots
                    if model._displayed_plots
                    else list(model.plots_map.keys())
                )
                if plots_to_use:
                    story.append(Paragraph("Figures", s_h2))
                    for plot_name in plots_to_use:
                        if plot_name not in model.plots_map:
                            continue
                        try:
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            fig = model.plots_map[plot_name](
                                backend="matplotlib"
                            )
                            figs = fig if isinstance(fig, list) else [fig]
                            for idx, f in enumerate(figs):
                                png = os.path.join(
                                    tmpdir,
                                    f"{model.model_id}_{plot_name}_{idx}.png"
                                )
                                f.savefig(png, bbox_inches='tight', dpi=150)
                                plt.close(f)
                                img_w = PAGE_W - 2*inch
                                story.append(Image(
                                    png,
                                    width=img_w,
                                    height=img_w * 0.6
                                ))
                                story.append(Spacer(1, 0.1*inch))
                        except Exception as e:
                            story.append(Paragraph(
                                f"[Figure '{plot_name}' unavailable: {e}]",
                                s_note
                            ))

                story.append(PageBreak())

            # ── Comparison section (multiple models) ──────────────────────
            if len(models) > 1:
                story.append(Paragraph("Model Comparison", s_h1))
                story.append(HRFlowable(
                    width="100%", thickness=0.5, color=GRAY, spaceAfter=6
                ))

                # Metrics
                all_keys = []
                for m in models:
                    if hasattr(m, 'metrics') and m.metrics:
                        for k, v in m.metrics.items():
                            if isinstance(v, (int, float)) and k not in all_keys:
                                all_keys.append(k)

                if all_keys:
                    story.append(Paragraph("Metrics", s_h2))
                    hdr = ["Metric"] + [m.model_id for m in models]
                    rows = [hdr]
                    for key in all_keys:
                        row = [key]
                        for m in models:
                            v = (m.metrics or {}).get(key, None)
                            row.append(
                                f"{v:.4f}" if isinstance(v, float)
                                else str(v) if v is not None else "—"
                            )
                        rows.append(row)
                    col_w = (PAGE_W - 2*inch) / len(hdr)
                    t = Table(rows, colWidths=[col_w] * len(hdr))
                    t.setStyle(_tbl_style())
                    story.append(t)
                    story.append(Spacer(1, 0.2*inch))

                # Common variables
                common_vars = None
                for m in models:
                    if hasattr(m, 'result_df') and m.result_df is not None:
                        vc = next(
                            (c for c in ['Variable', 'Feature', 'feature']
                             if c in m.result_df.columns), None
                        )
                        if vc:
                            vars_m = set(m.result_df[vc].tolist())
                            common_vars = (
                                vars_m if common_vars is None
                                else common_vars & vars_m
                            )

                if common_vars:
                    story.append(Paragraph(
                        f"Common Variables ({len(common_vars)} shared)", s_h2
                    ))
                    coef_cands = [
                        'Coefficient', 'OddsRatio', 'HazardRatio',
                        'Importance', 'Effect',
                    ]
                    hdr = ["Variable"] + [m.model_id for m in models]
                    rows = [hdr]
                    for var in sorted(common_vars):
                        row = [var]
                        for m in models:
                            if hasattr(m, 'result_df') and m.result_df is not None:
                                vc = next(
                                    (c for c in ['Variable', 'Feature', 'feature']
                                     if c in m.result_df.columns), None
                                )
                                cc = next(
                                    (c for c in coef_cands
                                     if c in m.result_df.columns), None
                                )
                                if vc and cc:
                                    match = m.result_df[m.result_df[vc] == var]
                                    v = match.iloc[0][cc] if not match.empty else None
                                    row.append(
                                        f"{v:.4f}" if isinstance(v, float)
                                        else str(v) if v is not None else "—"
                                    )
                                else:
                                    row.append("—")
                            else:
                                row.append("—")
                        rows.append(row)
                    col_w = (PAGE_W - 2*inch) / len(hdr)
                    t = Table(rows, colWidths=[col_w] * len(hdr))
                    t.setStyle(_tbl_style())
                    story.append(t)

            # ── Build ─────────────────────────────────────────────────────
            doc = SimpleDocTemplate(
                pdf_filename,
                pagesize=A4,
                rightMargin=inch,
                leftMargin=inch,
                topMargin=0.8*inch,
                bottomMargin=0.8*inch,
            )
            doc.build(
                story,
                onFirstPage=_hf,
                onLaterPages=_hf,
            )   

    # ======================================================================
    # ABSTRACT CONTRACT
    # ======================================================================

    @classmethod
    def create(
        cls,
        data: DataFrame,
        model: str,
        **params
    ) -> "RAPID":
        """
        Configure and instantiate the analytical pipeline.

        This method receives data and model configuration, validates
        the model type, and returns a configured pipeline instance
        ready for training.

        Args:
            data: Input DataFrame in ARC format.
            model: Model type identifier.
                Options: "logistic", "survival_cox", "survival_km", "glm",
                        "lca", "decision_tree", "random_forest", "xgboost",
                        "lightgbm", "catboost", "lasso", "ridge",
                        "elastic_net", "svm", "logistic_l2", "kmeans",
                        "descriptive".
            **params: Model-specific parameters.

        Returns:
            RAPID instance configured and ready for training.

        Raises:
            ValueError: If model type is unknown or required parameters missing.
        """
        # Imports lazy - evitam circular import
        from isaric.modeling.regression import LogisticRegression, GLM
        from isaric.modeling.survival import Survival
        from isaric.modeling.clustering import LCA, KMeans
        from isaric.modeling.descriptive import Descriptive
        from isaric.modeling.treebased import DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost
        from isaric.modeling.predictive import Lasso, Ridge, ElasticNet, SVM, LogisticL2

        # Mapeia string → classe
        registry = {
            "logistic": LogisticRegression,
            "glm": GLM,
            "survival": Survival,
            "lca": LCA,
            "kmeans": KMeans,
            "descriptive": Descriptive,
            "decision_tree": DecisionTree,
            "random_forest": RandomForest,
            "xgboost": XGBoost,
            "lightgbm": LightGBM,
            "catboost": CatBoost,
            "lasso": Lasso,
            "ridge": Ridge,
            "elastic_net": ElasticNet,
            "svm": SVM,
            "logistic_l2": LogisticL2,
        }

        # Valida
        if model not in registry:
            available_models = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Available models: {available_models}"
            )

        # Delega para a subclasse
        pipeline_cls = registry[model]
        instance = pipeline_cls.create(data=data, model=model, **params)

        # Registra bibliotecas usadas baseado no modelo
        instance._register_library("pandas")
        instance._register_library("numpy")

        if model in ["logistic", "glm"]:
            instance._register_library("statsmodels")
        elif model in ["survival_cox", "survival_km"]:
            instance._register_library("lifelines")
        elif model in ["lca"]:
            instance._register_library("stepmix")
        elif model in ["kmeans", "decision_tree", "random_forest", "lasso", 
                    "ridge", "elastic_net", "svm", "logistic_l2"]:
            instance._register_library("scikit-learn")
        elif model == "xgboost":
            instance._register_library("xgboost")
        elif model == "lightgbm":
            instance._register_library("lightgbm")
        elif model == "catboost":
            instance._register_library("catboost")

        # Gera model_id automático — sobrescrevível pelo usuário
        from datetime import datetime
        instance.model_id = (
            f"{model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        return instance


    def fit(
        self,
        metrics: Optional[List[str]] = None,
        cross_validation: bool = False,
        k_folds: int = 5,
        repetitions: int = 1,
        calibration: bool = False,
        assumptions: bool = False,
        train_test: bool = False,
        test_size: Optional[float] = None,
        grid_search: bool = False,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        selection_metric: str = "auto",
        bootstrap: bool = False,
        bootstrap_type: str = "auto",
        n_iterations: int = 1000,
        balancing: Optional[str] = None,
        threshold: Union[float, str] = 0.5,
        recalibrate: bool = False,
        recalibrate_on: str = "train",
    ) -> "RAPID":
        """
        Train the model and compute evaluation metrics.

        This method implements Steps 3 (Modelling) and 4 (Model Evaluation)
        of the RAPID methodology.

        Subclasses MUST have:
        - self._model (configured model from create)
        - self.X, self.y (data matrices)

        Args:
            metrics: List of performance metrics (None = defaults).
                Accepted Values:
                    - Classification: ["auc", "accuracy", "precision", "recall", "f1", "log_loss", "brier_score"]
                    - Regression: ["mse", "rmse", "mae", "r2", "adjusted_r2"]
                    - Survival: ["c_index"]
                    - Information: ["aic", "bic", "entropy"]
            cross_validation: Enable k-fold cross-validation.
            k_folds: Number of folds.
            repetitions: Number of repetitions for repeated k-fold.
            calibration: Enable calibration curve generation.
            assumptions: Enable assumption checking.
            train_test: Enable train/test split validation.
            test_size: Proportion for test set (default 0.2)
            grid_search: Enable grid search over hyperparameters.
                For LCA this parameter is ignored — class enumeration
                always runs. Pass param_grid to control the k range.
            param_grid: Dictionary mapping parameter names to list of values.
                If None, uses model-specific default.
            selection_metric: Metric to optimize during grid search.
                "auto" uses model-specific default.
            bootstrap: Enable bootstrap confidence intervals. When True,
                calls _run_bootstrap() after model training. Default False.
            bootstrap_type: Type of bootstrap CI to compute.
                Options: "percentile", "bias_corrected", "bca", "parametric",
                "auto" (uses model-specific default).
                - "percentile": simple quantile-based CI.
                - "bias_corrected": BC correction for skewed distributions.
                - "bca": bias-corrected and accelerated (computationally
                  expensive, standard for regression coefficients).
                - "parametric": samples from the fitted model's distribution
                  (LCA only, uses bootstrap_stepmix).
                Default "auto".
            n_iterations: Number of bootstrap samples. Only used when
                bootstrap=True. Default 1000.
            balancing: Class balancing applied to training data only.
                Options: None (no balancing, default), "undersample"
                (random undersampling of majority class), "smote"
                (synthetic minority oversampling). Only active for
                classification models. Requires imbalanced-learn.
            threshold: Classification threshold for converting predicted
                probabilities to binary predictions. Options:
                - float: fixed threshold (default 0.5).
                - "f1": selects threshold on training data that maximises
                  F1-score using a grid from 0.01 to 0.95 step 0.005.
                - "youden": selects threshold maximising Youden's J statistic.
                Only active for classification models.
            recalibrate: Apply logistic recalibration after training.
                Adjusts predicted probabilities to better match observed
                event rate. Does not change AUC or patient ranking.
                Only active for classification models. Default False.
            recalibrate_on: Data used to fit the recalibration model.
                - "train": complete training data before balancing (default,
                  matches paper method — avoids test set contamination).
                - "test": held-out test set (only when test_size is set).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If parameters are invalid or state is incorrect.
        """
        import warnings

        # Aceita CREATED (primeiro fit) ou FITTED (re-fit)
        if self._state == self._STATE_FITTED:
            warnings.warn(
                f"[{self.model_id}] Re-fitting model. "
                f"Previous fitted_model, metrics, result_df and "
                f"bootstrap_results will be overwritten. "
                f"To compare models, create a new instance with RAPID.create().",
                UserWarning,
                stacklevel=2,
            )
        elif self._state != self._STATE_CREATED:
            raise ValueError(
                f"Cannot call fit() in state '{self._state}'. "
                f"Required state: '{self._STATE_CREATED}' or '{self._STATE_FITTED}'."
            )

        from datetime import datetime
        from sklearn.model_selection import train_test_split as _sklearn_split

        # ── Store params ──────────────────────────────────────────────────
        self._bootstrap      = bootstrap
        self._bootstrap_type = bootstrap_type
        self._n_iterations   = n_iterations
        self._threshold      = threshold
        self._balancing      = balancing
        self._recalibrate    = recalibrate
        self._recalibrate_on = recalibrate_on

        # Auto-ativa grid_search se param_grid foi passado
        if not grid_search and param_grid is not None:
            grid_search = True

        # ── Stratified split (NEW — before training) ──────────────────────
        _is_classifier = self.model_type in self._CLASSIFICATION_MODELS
        _do_new_split  = test_size is not None

        if _do_new_split:
            X_arr = self.X
            y_arr = self.y

            # Stratified for classifiers, random for others
            stratify_col = y_arr if _is_classifier else None
            X_tr, X_te, y_tr, y_te = _sklearn_split(
                X_arr, y_arr,
                test_size=test_size,
                random_state=self._random_seed,
                stratify=stratify_col,
            )

            # Keep full training data for recalibration (before balancing)
            self.X_train_full = X_tr
            self.y_train_full = y_tr

            # Apply balancing to training data only
            if balancing and _is_classifier:
                X_tr, y_tr = self._apply_balancing(
                    X_tr, y_tr,
                    method=balancing,
                    random_state=self._random_seed,
                )

            self.X_train = X_tr
            self.y_train = y_tr
            self.X_test  = X_te
            self.y_test  = y_te

        else:
            # No new split — use all data (backward compatible)
            self.X_train      = self.X
            self.y_train      = self.y
            self.X_train_full = self.X
            self.y_train_full = self.y
            self.X_test       = None
            self.y_test       = None

        # ── Step 3: Train model ───────────────────────────────────────────
        self.fitted_model = self._train_model(
            grid_search=grid_search,
            param_grid=param_grid,
            selection_metric=selection_metric,
        )

        self._fit_at = datetime.now().isoformat()

        # ── Build result_df ───────────────────────────────────────────────
        self.result_df = self._build_result_df()

        # ── Step 4: Calculate metrics ─────────────────────────────────────
        self.metrics = self._calculate_metrics(metrics)

        # ── Recalibration (after metrics) ─────────────────────────────────
        if recalibrate and _is_classifier and self.fitted_model is not None:
            if recalibrate_on == "train" and self.X_train_full is not None:
                cal_results = self._apply_recalibration(
                    self.fitted_model,
                    self.X_train_full,
                    self.y_train_full,
                )
            elif recalibrate_on == "test" and self.X_test is not None:
                cal_results = self._apply_recalibration(
                    self.fitted_model,
                    self.X_test,
                    self.y_test,
                )
            else:
                cal_results = {}

            if cal_results:
                self._recalibration_slope     = cal_results.get('calibration_slope')
                self._recalibration_intercept = cal_results.get('calibration_intercept')
                if self.metrics is None:
                    self.metrics = {}
                self.metrics.update(cal_results)

        # ── Cross-validation ──────────────────────────────────────────────
        if cross_validation:
            self.cv_metrics = self._cross_validate(k_folds, repetitions)

        # ── Calibration ───────────────────────────────────────────────────
        if calibration:
            self.calibration = self._calibration_curve()

        # ── Assumptions ───────────────────────────────────────────────────
        if assumptions:
            self.assumptions = self._check_assumptions()

        # ── Train/Test (OLD mechanism — backward compat) ──────────────────
        if train_test and not _do_new_split:
            self.train_test = self._train_test_split(
                test_size if test_size is not None else 0.2
            )

        # ── Bootstrap CIs ─────────────────────────────────────────────────
        if bootstrap:
            self.bootstrap_results = self._validate_bootstrap(n_iterations)

        self._transition_to(self._STATE_FITTED)
        return self

    
    def summary(
        self,
        table_format: str = "full",
        plots: Optional[List[str]] = None,
        compare: Optional[List["RAPID"]] = None,
        reorder: Optional[Dict] = None,
    ) -> None:
        """
        Display results in tabular and graphical format.

        When called without compare=, displays this instance's results.
        When called with compare=[m2, m3], displays a comparative table
        with self as reference (first column) against the listed models.
        All models must be from the same family.

        Args:
            table_format: "full" or "short". Ignored in compare mode.
            plots: List of plot names. Ignored in compare mode.
            compare: Optional list of additional RAPID instances to
                compare against self. Self is always the reference.
                All models must be fitted and from the same family.

        Raises:
            ValueError: If state is incorrect, models are from different
                families, or any compared model is not fitted.
        """
        
        # ── Comparative mode ──────────────────────────────────────────────
        if compare is not None:
            all_models = [self] + list(compare)

            # Validate all models are fitted
            not_fitted = [
                m.model_id for m in all_models
                if m._state != self._STATE_FITTED
            ]
            if not_fitted:
                raise ValueError(
                    f"All models must be fitted before comparison. "
                    f"Not fitted: {not_fitted}. Call fit() first."
                )

            # Validate same family
            families = {
                self._FAMILY_MAP.get(m.model_type, 'unknown')
                for m in all_models
            }
            if len(families) > 1:
                details = [
                    f"{m.model_id} ({m.model_type})"
                    for m in all_models
                ]
                raise ValueError(
                    f"summary(compare=[...]) requires all models to be "
                    f"from the same family. "
                    f"Found families: {sorted(families)}. "
                    f"Models: {details}."
                )

            self._summary_comparative(all_models)
            return

        # Store reorder mapping for display (plots and tables)
        if reorder is not None:
            self._display_reorder = reorder

        # Store displayed plots for report()
        # Priority: explicit plots > model _default_plots > all plots_map keys
        if plots is not None:
            self._displayed_plots = plots
        elif hasattr(self, '_default_plots') and self._default_plots:
            self._displayed_plots = [
                p for p in self._default_plots
                if p in self.plots_map
            ]
        else:
            self._displayed_plots = list(self.plots_map.keys())

        # ======================================================================
        # 1. MODEL INFORMATION
        # ======================================================================
        if hasattr(self, 'model_info') and self.model_info:
            print("=" * 80)
            print("MODEL INFORMATION")
            print("=" * 80)

            # Converte dict para DataFrame para exibição uniforme
            import pandas as pd
            info_df = pd.DataFrame(
                list(self.model_info.items()),
                columns=["Key", "Value"]
            )

            # Formata valores: float → 4 casas decimais, outros → str
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            info_df["Value"] = info_df["Value"].apply(_fmt)
            print(info_df.to_string(index=False))
            print()

        # ======================================================================
        # 2. RESULTS
        # ======================================================================
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)

        if table_format == "full":
            print(self.result_df.to_string(index=False))
        elif table_format == "short":
            # Short: ratio columns take priority over raw Coefficient.
            # Shows: Variable + main effect + CI + p-value.
            ratio_cols = ['OddsRatio', 'HazardRatio', 'MeanRatio']
            has_ratio  = any(c in self.result_df.columns for c in ratio_cols)

            priority = ['Variable']
            priority += [c for c in ratio_cols if c in self.result_df.columns]
            if not has_ratio and 'Coefficient' in self.result_df.columns:
                priority.append('Coefficient')
            if 'Importance' in self.result_df.columns:
                priority.append('Importance')
            for c in ['LowerCI', 'UpperCI', 'p-value']:
                if c in self.result_df.columns:
                    priority.append(c)

            display_cols = [c for c in priority if c in self.result_df.columns]
            print(
                self.result_df[display_cols].to_string(index=False)
                if display_cols
                else self.result_df.to_string(index=False)
            )
        else:
            raise ValueError(
                f"table_format must be 'full' or 'short'. "
                f"Received: '{table_format}'"
            )
        print()

        # ======================================================================
        # 2b. BOOTSTRAP CONFIDENCE INTERVALS (when bootstrap=True in fit())
        # ======================================================================
        if (hasattr(self, 'bootstrap_results')
                and self.bootstrap_results is not None):
            btype = self.bootstrap_results.get('bootstrap_type', 'unknown')
            n_it  = self.bootstrap_results.get('n_iterations', '?')
            print("=" * 80)
            print(
                f"BOOTSTRAP CONFIDENCE INTERVALS"
                f" — {btype.upper()} (n={n_it})"
            )
            print("=" * 80)

            # Regression / Predictive — coefficients
            # Omits bootstrap_mean and bootstrap_se (confusing, Rec 10).
            # Shows OR CIs when available (Rec 11).
            boot_df = self.bootstrap_results.get('coefficients')
            if boot_df is not None and not boot_df.empty:
                # Prefer OR scale when available (logistic); fall back to coef scale
                if 'odds_ratio' in boot_df.columns:
                    show_cols = [
                        c for c in [
                            'variable', 'odds_ratio',
                            'or_ci_lower', 'or_ci_upper',
                            'bootstrap_p_value',
                        ]
                        if c in boot_df.columns
                    ]
                else:
                    show_cols = [
                        c for c in [
                            'variable', 'observed',
                            'ci_lower', 'ci_upper',
                            'bootstrap_p_value',
                        ]
                        if c in boot_df.columns
                    ]
                print(boot_df[show_cols].to_string(index=False))

            # Survival — hazard ratios
            elif self.bootstrap_results.get('hazard_ratios') is not None:
                boot_df = self.bootstrap_results['hazard_ratios']
                show_cols = [
                    c for c in [
                        'variable', 'observed_hr', 'hr_ci_lower',
                        'hr_ci_upper', 'bootstrap_p_value',
                    ]
                    if c in boot_df.columns
                ]
                print(boot_df[show_cols].to_string(index=False))

            # Tree-based — feature importances
            elif self.bootstrap_results.get('importances') is not None:
                boot_df = self.bootstrap_results['importances']
                show_cols = [
                    c for c in [
                        'Feature', 'observed_importance',
                        'ci_lower', 'ci_upper',
                    ]
                    if c in boot_df.columns
                ]
                print(boot_df[show_cols].to_string(index=False))

            print()

        # ======================================================================
        # 3. GRID SEARCH RESULTS
        # ======================================================================
        if hasattr(self, 'grid_results') and self.grid_results is not None:
            print("=" * 80)
            print("GRID SEARCH RESULTS")
            print("=" * 80)
            print(self.grid_results.to_string(index=False))

            if hasattr(self, 'best_params') and self.best_params:
                print(f"\nBest parameters: {self.best_params}")
            print()

        # ======================================================================
        # 3b. OUTCOME CROSSTAB (when outcome was provided in create())
        # ======================================================================
        if (hasattr(self, 'outcome_crosstab_df')
                and self.outcome_crosstab_df is not None):
            outcome_name = getattr(self, 'outcome', 'outcome')
            print("=" * 80)
            print(f"OUTCOME BY CLASS — {outcome_name}")
            print("=" * 80)
            print(self.outcome_crosstab_df.to_string())
            if (hasattr(self, 'outcome_test_result')
                    and self.outcome_test_result):
                t = self.outcome_test_result
                print(
                    f"\n  {t.get('test', '')}: "
                    f"statistic={t.get('statistic', float('nan')):.4f}, "
                    f"p-value={t.get('p_value', float('nan')):.4f}"
                )
            print()

        # ======================================================================
        # 3c. LOG-RANK TEST (when group_var was provided in create())
        # ======================================================================
        if (hasattr(self, 'logrank_results')
                and self.logrank_results is not None
                and 'error' not in self.logrank_results):
            print("=" * 80)
            print("LOG-RANK TEST")
            print("=" * 80)
            groups = self.logrank_results.get('groups', [])
            stat   = self.logrank_results.get('test_statistic', float('nan'))
            pval   = self.logrank_results.get('p_value', float('nan'))
            print(f"  Groups compared : {', '.join(str(g) for g in groups)}")
            print(f"  Test statistic  : {stat:.4f}")
            pval_str = f"{pval:.4f}" if pval >= 0.001 else f"{pval:.2e}"
            print(f"  p-value         : {pval_str}")
            if pval < 0.05:
                print(f"  Interpretation  : Significant difference in survival"
                      f" between groups (p < 0.05)")
            else:
                print(f"  Interpretation  : No significant difference in survival"
                      f" between groups (p ≥ 0.05)")
            print()

        # ======================================================================
        # 4. PLOTS
        # ======================================================================
        if plots:
            for plot in plots:
                if plot in self.plots_map:
                    fig = self.plots_map[plot]()
                    if isinstance(fig, list):
                        for f in fig:
                            f.show()
                    else:
                        fig.show()
                else:
                    raise ValueError(f"Unknown plot: {plot}")

    def save(self) -> "RAPID":
        """
        Persist the trained model to disk.

        Saves a ZIP file containing:
        - metadata.json: model name, type, model_id, creation timestamp,
          RAPID version, library versions, key metrics, validation log,
          and decision (if decide() was called).
        - model.pkl: serialized fitted model (pickle).

        Can be called any time after fit(). Does not require summary()
        to have been called first.

        Subclasses MUST have:
        - self.model_type (str)
        - self.fitted_model (trained model)
        - self.metrics (dict with performance metrics)

        Returns:
            self for method chaining.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        self._check_state(self._STATE_FITTED, "save")

        import json
        import pickle
        import zipfile
        from datetime import datetime
        import isaric

        filename = (
            f"{self.model_id}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.rapid"
        )

        # Serialise validation log — convert DataFrames to dicts
        validation_log_serialisable = []
        for entry in self._validation_log:
            entry_copy = {}
            for k, v in entry.items():
                try:
                    import pandas as pd
                    if isinstance(v, pd.DataFrame):
                        entry_copy[k] = v.to_dict(orient='records')
                    elif isinstance(v, dict):
                        entry_copy[k] = {
                            str(kk): (
                                vv.to_dict(orient='records')
                                if isinstance(vv, pd.DataFrame)
                                else vv
                            )
                            for kk, vv in v.items()
                        }
                    else:
                        entry_copy[k] = v
                except Exception:
                    entry_copy[k] = str(v)
            validation_log_serialisable.append(entry_copy)

        metadata = {
            "model_id":        self.model_id,
            "model_name":      self.model_type,
            "model_type":      self.model_type,
            "created_at":      datetime.now().isoformat(),
            "fit_at":          self._fit_at,
            "rapid_version":   isaric.__version__,
            "library_versions": self._libraries_used,
            "key_metrics":     self.metrics,
            "validation_log":  validation_log_serialisable,
            "decision":        self._decision,
        }

        with zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('metadata.json', json.dumps(metadata, indent=2))
            zf.writestr('model.pkl', pickle.dumps(self.fitted_model))

        self.saved_filename = filename
        print(f"✅ Model saved: {filename}")
        return self

    def validation(
        self,
        external_data: Optional[DataFrame] = None,
        sensitivity: bool = False,
        subgroups: Optional[Dict] = None,
        net_benefit: bool = False
    ) -> Dict:
        """
        Validate the trained model using independent external data.

        Reserved for performance assessment on independent datasets —
        analogous to a TEST set in ML (TRAIN / VALIDATION / TEST split).

        Each call is accumulated in self._validation_log, allowing
        multiple validations (different datasets, different parameters)
        to be recorded and included in report().

        Bootstrap confidence intervals are computed during fit()
        via bootstrap=True, not here.

        Can be called any time after fit(). Does not require summary()
        to have been called first.

        Subclasses MUST have:
        - self.fitted_model (trained model)

        Args:
            external_data: Independent dataset for external validation
                (e.g. data from another site, a held-out time period,
                or data collected subsequently).
            sensitivity: Enable sensitivity analysis.
            subgroups: Dictionary defining subgroups for stratified
                performance assessment.
            net_benefit: Enable Decision Curve Analysis (DCA).

        Returns:
            Dictionary with validation results for this call.

        Raises:
            ValueError: If model has not been fitted yet.
        """
        self._check_state(self._STATE_FITTED, "validation")

        from datetime import datetime

        validation_results = {}

        # External validation — independent dataset
        if external_data is not None:
            validation_results['external'] = self._validate_external(
                external_data
            )

        # Sensitivity
        if sensitivity:
            validation_results['sensitivity'] = self._validate_sensitivity()

        # Subgroups
        if subgroups:
            validation_results['subgroups'] = self._validate_subgroups(
                subgroups
            )

        # Net benefit
        if net_benefit:
            validation_results['net_benefit'] = self._validate_net_benefit()

        # Accumulate in validation log
        log_entry = {
            'timestamp':     datetime.now().isoformat(),
            'external_data': str(type(external_data).__name__) if external_data is not None else None,
            'sensitivity':   sensitivity,
            'subgroups':     str(subgroups) if subgroups else None,
            'net_benefit':   net_benefit,
            'results':       validation_results,
        }
        self._validation_log.append(log_entry)

        return validation_results

    def report(
        self,
        models: Optional[List["RAPID"]] = None,
        format: Optional[List[str]] = None,
    ) -> None:
        """
        Generate a publication-ready report for one or more fitted models.

        Can be called as a single-model report without arguments:
            model.report()
            model.report(format=["pdf", "csv"])

        Or as a multi-model consolidated report:
            model.report(models=[model, model2], format=["pdf"])

        When models=None (default), reports self only.
        When models=[...] is provided, reports exactly those models.

        CSV and PNG exports are generated per model. The PDF is
        consolidated across all models.

        Args:
            models: List of fitted RAPID instances to include.
                Single model: RAPID.report(models=[model])
                Multiple:     RAPID.report(models=[m1, m2, m3])
            format: List of output formats. Options: "pdf", "csv", "png".
                Default ["pdf"].

        Raises:
            ValueError: If any model is not fitted, models list is empty,
                or models are from different families.
        """
        import os
        import isaric
        from datetime import datetime

        # Default: report self only
        if models is None:
            models = [self]
        elif not models:
            raise ValueError(
                "models list is empty. "
                "Pass at least one fitted RAPID instance."
            )

        # Validate all models are fitted
        not_fitted = [
            m.model_id for m in models
            if m._state != self._STATE_FITTED
        ]
        if not_fitted:
            raise ValueError(
                f"All models must be fitted before report(). "
                f"Not fitted: {not_fitted}. Call fit() first."
            )

        # Validate same family when multiple models
        if len(models) > 1:
            families = {
                self._FAMILY_MAP.get(m.model_type, 'unknown')
                for m in models
            }
            if len(families) > 1:
                details = [
                    f"{m.model_id} ({m.model_type})" for m in models
                ]
                raise ValueError(
                    f"report() requires all models to be from the same "
                    f"family when comparing. "
                    f"Found families: {sorted(families)}. "
                    f"Models: {details}."
                )

        format_list = format if format else ["pdf"]
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        # ── CSV export — one file per model ──────────────────────────────
        if "csv" in format_list:
            for model in models:
                prefix = f"{model.model_id}-{timestamp}"

                def _add_context(df) -> 'DataFrame':
                    """Add model context columns to exported DataFrame."""
                    df = df.reset_index(drop=True).copy()
                    df.insert(0, 'exported_at', timestamp)
                    df.insert(0, 'model_type',  model.model_type)
                    df.insert(0, 'model_id',    model.model_id)
                    return df

                if hasattr(model, 'result_df') and model.result_df is not None:
                    csv_file = f"{prefix}_results.csv"
                    _add_context(model.result_df).to_csv(csv_file, index=False)
                    print(f"✅ CSV saved: {csv_file}")

                if hasattr(model, 'grid_results') and model.grid_results is not None:
                    csv_grid = f"{prefix}_grid_results.csv"
                    _add_context(model.grid_results).to_csv(csv_grid, index=False)
                    print(f"✅ CSV saved: {csv_grid}")

                # Additional exports for LCA
                if (hasattr(model, 'outcome_crosstab_df')
                        and model.outcome_crosstab_df is not None):
                    csv_ct = f"{prefix}_outcome_crosstab.csv"
                    _add_context(
                        model.outcome_crosstab_df.reset_index()
                    ).to_csv(csv_ct, index=False)
                    print(f"✅ CSV saved: {csv_ct}")

                if hasattr(model, 'DiagClass') and model.DiagClass:
                    ave_pp = model.DiagClass.get('ave_pp')
                    if ave_pp is not None and not ave_pp.empty:
                        csv_pp = f"{prefix}_ave_pp.csv"
                        _add_context(
                            ave_pp.reset_index()
                        ).to_csv(csv_pp, index=False)
                        print(f"✅ CSV saved: {csv_pp}")

        # ── PNG export — one file per model per plot ──────────────────────
        if "png" in format_list:
            for model in models:
                prefix = f"{model.model_id}-{timestamp}"
                plots_to_export = (
                    model._displayed_plots
                    if model._displayed_plots
                    else list(model.plots_map.keys())
                )
                for plot_name in plots_to_export:
                    if plot_name not in model.plots_map:
                        continue
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        fig = model.plots_map[plot_name](backend="matplotlib")
                        figs = fig if isinstance(fig, list) else [fig]
                        for idx, f in enumerate(figs):
                            png_file = f"{prefix}_{plot_name}_{idx}.png"
                            f.savefig(png_file, bbox_inches='tight', dpi=150)
                            import matplotlib.pyplot as plt
                            plt.close(f)
                            print(f"✅ PNG saved: {png_file}")
                    except Exception as e:
                        print(f"⚠️  Plot '{plot_name}' skipped: {e}")

        # ── PDF export — consolidated across all models ───────────────────
        if "pdf" in format_list:
            model_ids = "_".join(m.model_id for m in models[:3])
            if len(models) > 3:
                model_ids += f"_and_{len(models) - 3}_more"
            pdf_filename = f"RAPID_report_{model_ids}-{timestamp}.pdf"
            self._generate_pdf_report(models, pdf_filename)
            print(f"✅ PDF saved: {pdf_filename}")

    def decide(
        self,
        selected: bool = True,
        reason: Optional[str] = None,
        k: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> "RAPID":
        """
        Document the model selection decision for scientific traceability.

        Records whether this model was selected, the reason for the
        decision, and any additional notes. The decision is stored in
        self._decision and included in save() metadata and report() PDF.

        This method is about documentation and reproducibility, not
        about controlling which files go into a report. The report
        is controlled by passing models to RAPID.report(models=[...]).

        Can be called any time after fit(). Calling decide() multiple
        times overwrites the previous decision.

        For LCA: pass k= to document which number of classes was chosen
        after reviewing class enumeration results.

        Args:
            selected: Whether this model was selected for the final
                analysis. Default True.
            reason: Scientific rationale for the selection or rejection.
                Good practice to include metrics, clinical considerations,
                and validation results that informed the decision.
                Example: "k=3 selected. BIC elbow at k=3, entropy=0.82,
                AvePP > 0.85 across all classes. Clinically validated
                with infectiologist team on 2026-10-21."
            k: For LCA only — the number of classes selected after
                reviewing class enumeration results. Has no effect on
                other model types.
            notes: Optional free-text notes for additional context
                (caveats, limitations, follow-up actions, etc.).

        Returns:
            self for method chaining.

        Raises:
            ValueError: If model has not been fitted yet.

        Example:
            # Single model
            model.decide(
                selected=True,
                reason="Best AUC (0.847) with fewest variables. "
                       "Externally validated in Centro A (AUC 0.821)."
            )

            # LCA — document chosen k
            lca.decide(
                selected=True,
                k=3,
                reason="k=3: entropy=0.82, all AvePP > 0.85, "
                       "clinically interpretable profiles."
            )

            # Rejected model — document why
            model_b.decide(
                selected=False,
                reason="Lower AUC than model_a. Higher complexity "
                       "without clinical benefit."
            )
        """
        self._check_state(self._STATE_FITTED, "decide")

        from datetime import datetime

        # Validate k for LCA
        if k is not None:
            if self.model_type not in ('lca', 'kmeans'):
                import warnings
                warnings.warn(
                    f"k= is only meaningful for clustering models "
                    f"(lca, kmeans). Ignored for model_type='{self.model_type}'.",
                    UserWarning,
                    stacklevel=2,
                )
                k = None
            elif (hasattr(self, 'model_objects')
                  and self.model_objects
                  and k not in self.model_objects):
                raise ValueError(
                    f"k={k} was not in the class enumeration range. "
                    f"Available k: {sorted(self.model_objects.keys())}. "
                    f"Re-run fit() with a param_grid that includes k={k}."
                )

        self._decision = {
            'selected':   bool(selected),
            'reason':     reason,
            'k':          k,
            'notes':      notes,
            'timestamp':  datetime.now().isoformat(),
            'model_id':   self.model_id,
            'model_type': self.model_type,
        }

        # Confirmation message
        status = "✅ SELECTED" if selected else "❌ NOT SELECTED"
        print(f"{status} — {self.model_id}")
        if k is not None:
            print(f"  k selected: {k}")
        if reason:
            print(f"  Reason: {reason}")
        if notes:
            print(f"  Notes: {notes}")
        print(
            f"  Decision recorded at {self._decision['timestamp']}. "
            f"Included in save() and RAPID.report()."
        )

        return self