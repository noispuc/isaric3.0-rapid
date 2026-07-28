import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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


def compute_extended_classification_metrics(y_true, y_pred, y_proba=None):
    """
    Description:
        Calcula métricas de discriminação estendidas para classificação
        binária, complementando compute_classification_metrics com métricas
        relevantes para alvos raros/desbalanceados (ex.: hospitalização em
        dengue, prevalência ~5%): AUC-PR, NPV e Specificity.

    Args:
        y_true (array-like): Rótulos verdadeiros (0/1).
        y_pred (array-like): Rótulos previstos (0/1), já aplicado o
            threshold de decisão escolhido (ver select_classification_threshold).
        y_proba (array-like, optional): Probabilidades previstas da classe
            positiva, usadas para AUC-ROC, AUC-PR e Brier Score.

    Returns:
        dict: accuracy, precision, recall, f1, specificity, npv,
            confusion_matrix e, se y_proba for informado, auc_roc, auc_pr,
            brier_score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "npv": npv,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }

    if y_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        metrics["auc_pr"] = average_precision_score(y_true, y_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)

    return metrics


def select_classification_threshold(y_true, y_proba, method="f1"):
    """
    Description:
        Seleciona um threshold de classificação a partir de probabilidades
        preditas, no lugar do corte padrão de 0.5, necessário quando o alvo
        é raro (ex.: hospitalização em dengue). Deve ser calculado somente
        com dados de treino/validação (fold-safe), nunca com o conjunto de
        teste — caso contrário o threshold vaza informação do teste.

    Args:
        y_true (array-like): Rótulos verdadeiros (0/1) do bloco de treino/validação.
        y_proba (array-like): Probabilidades previstas da classe positiva.
        method (str): 'f1' maximiza o F1-score; 'youden' maximiza o índice
            de Youden (sensibilidade + especificidade - 1).

    Returns:
        float: Threshold de decisão selecionado.
    """
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        if len(thresholds) == 0:
            return 0.5
        denom = precision[:-1] + recall[:-1]
        f1_scores = np.divide(
            2 * precision[:-1] * recall[:-1], denom, out=np.zeros_like(denom), where=denom > 0
        )
        return float(thresholds[np.argmax(f1_scores)])

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden = tpr - fpr
        return float(thresholds[np.argmax(youden)])

    raise ValueError(f"method inválido: '{method}'. Use 'f1' ou 'youden'.")
