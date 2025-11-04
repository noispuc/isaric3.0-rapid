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
