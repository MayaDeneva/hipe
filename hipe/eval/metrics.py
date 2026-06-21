from sklearn.metrics import recall_score, confusion_matrix


def macro_recall(y_true, y_pred) -> float:
    labels = sorted(set(y_true))
    return float(recall_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))


def confusion(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    return labels, confusion_matrix(y_true, y_pred, labels=labels)
