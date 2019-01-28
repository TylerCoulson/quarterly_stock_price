import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

def normalize_confusion_matrix(confusion_matrix,axis=1):
    """
    This function returns a confusion matrix that has been normalized on a percentage scale.

    Parameters
    ----------

    confusion_matrix: array, shape = [n_classes, n_classes]
        The result of sklearn.metrics.confusion_matrix

    axis: int, default 1
        The axis used for normalization
        0 normalize by column, 1 normalize by row

    Returns
    -------
    normalized_confusion_matrix: array, shape = [n_classes, n_classes]
        normalized confusion matrix
    """

    axis_total = confusion_matrix.sum(axis=axis)
    axis_total[axis_total==0] = 1

    if axis == 1:
    # np.newaxis reshapes axis_total to a 1-col x n-row array for row division
        return confusion_matrix/(axis_total[:,np.newaxis])
    else:
        return confusion_matrix/(axis_total)

def confusion_heatmap(y_test, y_pred, labels=None, sample_weight=None, normalize=False):

    """
    Creates a it's seaborn heatmap

     Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with the heatmap.

    """
    labels = np.unique(y_pred) if labels == None else labels

    cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)

    normalize_dict = {
        False: cm,
        True: normalize_confusion_matrix(confusion_matrix=cm, axis=1),
        'predicted_values': normalize_confusion_matrix(confusion_matrix=cm, axis=0),
        'precision': normalize_confusion_matrix(confusion_matrix=cm, axis=0),
        'true_values': normalize_confusion_matrix(confusion_matrix=cm, axis=1),
        'accuracy': normalize_confusion_matrix(confusion_matrix=cm, axis=1),
        }
    cm = normalize_dict[normalize]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    cm_df = cm_df.rename_axis('true_labels',axis=0).rename_axis('predicted_labels',axis=1)

    cm_plot = sns.heatmap(cm_df, cmap='Blues', annot=True, fmt='.3f', linewidths=.2, vmin=0)
    cm_plot.figure.set_size_inches(12, 8)
    cm_plot.set_xticklabels(cm_plot.get_xticklabels(), rotation = 0,)

    return cm_plot
