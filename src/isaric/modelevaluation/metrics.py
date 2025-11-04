from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Description:
        Computes common classification metrics including accuracy, AUC, and confusion matrix.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted class labels.
        y_proba (array-like, optional): Predicted probabilities for AUC calculation.

    Returns:
        dict: Dictionary with accuracy, AUC (if available), and confusion matrix.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    if y_proba is not None:
        metrics["auc"] = roc_auc_score(y_true, y_proba)
    return metrics
