"""
Evaluation metrics, kept separate from train.py so the same reporting logic
can be reused to re-evaluate a saved model against new data.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from utils import logger


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_report(metrics: dict, title: str = "Evaluation Report"):
    logger.info(f"===== {title} =====")
    logger.info(f"Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall   : {metrics['recall']:.4f}")
    logger.info(f"F1 Score : {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    logger.info("Confusion Matrix (rows=true [real,screen], cols=pred [real,screen]):")
    logger.info(f"  {cm.tolist()}")
