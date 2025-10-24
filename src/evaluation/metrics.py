"""
Metrics calculation utilities for model evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Tuple, Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }


def calculate_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    class_accuracies = []

    for i in range(num_classes):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_correct = np.sum((y_pred[class_mask] == i))
            class_total = np.sum(class_mask)
            class_acc = class_correct / class_total
        else:
            class_acc = 0.0

        class_accuracies.append(class_acc)

    return np.array(class_accuracies)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> str:
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )


def analyze_errors(cm: np.ndarray, class_names: list, top_k: int = 10) -> list:
    mistakes = []

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                mistakes.append((int(cm[i][j]), class_names[i], class_names[j]))

    mistakes.sort(reverse=True)
    return mistakes[:top_k]