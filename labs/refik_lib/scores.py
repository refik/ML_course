import numpy as np


def compute_accuracy(pred_y, true_y):
    return np.mean(pred_y == true_y)


def compute_f1_score(y_true, y_pred):
    nb_positive = np.sum(y_pred == 1)
    nb_true_positive = np.sum(y_true == 1)
    if not(nb_positive and nb_true_positive):
        return 1.0
    # print(nb_positive, nb_true_positive)
    precision = np.sum((y_pred == y_true) & (y_true == 1)) / nb_positive
    recall = np.sum((y_pred == y_true) & (y_true == 1)) / nb_true_positive
    # print(precision, recall)

    return 2 * (precision * recall) / (precision + recall)


def compute_precision(y_true, y_pred):
    nb_positive = np.sum(y_pred == 1)
    if not nb_positive:
        return 1.0
    return np.sum((y_pred == y_true) & (y_true == 1)) / nb_positive


def compute_recall(y_true, y_pred):
    nb_true_positive = np.sum(y_true == 1)
    if not nb_true_positive:
        return 1.0
    return np.sum((y_pred == y_true) & (y_true == 1)) / nb_true_positive
