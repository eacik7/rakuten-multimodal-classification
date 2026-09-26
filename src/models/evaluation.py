import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)

def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Calcule l'ensemble des métriques de classification multiclasse.
    Métrique principale : Weighted F1-Score (officiel Challenge Rakuten ENS).
    """
    metrics = {
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_prob)
        except Exception:
            metrics["log_loss"] = np.nan
    return metrics

def get_detailed_report(y_true, y_pred, target_names=None):
    """
    Génère un DataFrame avec precision, recall, f1-score et support pour chaque catégorie.
    """
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    df_report = pd.DataFrame(report_dict).transpose()
    return df_report

def get_normalized_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcule la matrice de confusion normalisée par les vraies classes (taux de rappel).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return cm

def find_top_confusions(cm, class_names, top_k=12):
    """
    Identifie les paires de classes les plus fréquemment confondues (hors diagonale).
    """
    confusions = []
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                error_rate = cm[i, j]
                if error_rate > 0.005:  # au moins 0.5% d'erreur sur cette classe
                    confusions.append({
                        "true_class": class_names[i],
                        "pred_class": class_names[j],
                        "error_rate": error_rate
                    })
    df_conf = pd.DataFrame(confusions)
    if not df_conf.empty:
        df_conf = df_conf.sort_values(by="error_rate", ascending=False).head(top_k).reset_index(drop=True)
    return df_conf
