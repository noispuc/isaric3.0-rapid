def calculate_net_profit(y_true, y_pred, profit_matrix):
    """
    Description:
        Calculates net profit based on predictions and a custom profit matrix.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        profit_matrix (dict): Dictionary with keys ('TP', 'FP', 'TN', 'FN') and associated profit values.

    Returns:
        float: Total net profit.
    """
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (
        tp * profit_matrix.get("TP", 0) +
        fp * profit_matrix.get("FP", 0) +
        tn * profit_matrix.get("TN", 0) +
        fn * profit_matrix.get("FN", 0)
    )


def decision_curve_analysis(y_true, y_proba, thresholds=None):
    """
    Description:
        Decision Curve Analysis (DCA): net benefit of using the model across a
        range of threshold probabilities, compared with the 'treat all' and
        'treat none' reference strategies.

        Net benefit at threshold t is TP/n - (FP/n) * (t / (1 - t)), where the
        odds term converts false positives into the units of true positives,
        encoding how much harm a false positive causes relative to the benefit
        of a true positive at that threshold.

        This is distinct from `calculate_net_profit`, which evaluates a single
        operating point against an explicit cost matrix. DCA is what the RAPID
        package contract (HUB-BR-005-01, section 7.9) refers to as
        `_net_benefit_analysis`.

    Args:
        y_true (array-like): True binary labels (0/1).
        y_proba (array-like): Predicted probabilities for the positive class.
        thresholds (array-like, optional): Threshold probabilities to evaluate.
            Defaults to 0.01..0.99 in steps of 0.01.

    Returns:
        pandas.DataFrame: One row per threshold, with columns 'threshold',
        'net_benefit_model', 'net_benefit_treat_all' and
        'net_benefit_treat_none'. The output is aggregated: it carries no
        patient-level information.
    """
    import numpy as np
    import pandas as pd

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.shape != y_proba.shape:
        raise ValueError(
            f"y_true {y_true.shape} and y_proba {y_proba.shape} must have the same shape."
        )

    n = y_true.size
    if n == 0:
        raise ValueError("y_true must not be empty.")

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    thresholds = np.asarray(thresholds, dtype=float)

    prevalence = y_true.mean()
    rows = []
    for t in thresholds:
        odds = t / (1.0 - t)
        predicted = y_proba >= t
        tp = np.sum(predicted & (y_true == 1))
        fp = np.sum(predicted & (y_true == 0))
        rows.append({
            "threshold": float(t),
            "net_benefit_model": float(tp / n - (fp / n) * odds),
            "net_benefit_treat_all": float(prevalence - (1.0 - prevalence) * odds),
            "net_benefit_treat_none": 0.0,
        })

    return pd.DataFrame(rows)
