import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.figure import Figure

from __general_utils__ import decode_predictions


def plot_roc_auc(model: object, X_test: object, y_test: object) -> object:
    """
        Function to create a plot for the AUC and ROC curves.
        It takes the labels (classes), gets the probability metrics,
        and calculates the AUC ROC curve for all.
        :param model:
        :param X_test:
        :param y_test:
        :return: fig
    """

    classes = np.unique(y_test)  # get unique classes
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = model.predict_proba(X_test)  # gt probability scores

    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    for i in range(y_test_bin.shape[1]):
        if np.sum(y_test_bin[:, i]) == 0:  # Skip classes without positive samples
            print(f"Skipping class {classes[i]} (no positive samples)")
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class ROC Curve")
    ax.legend()

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')

    fig.tight_layout()

    return fig


def construct_confusion_matrix_visual(model: object, X_test: object, y_test: object, label_mappings: object, model_name: object) -> object:
    """
    Function to create and return a confusion matrix plot.
    :param model:
    :param X_test:
    :param y_test:
    :param label_mappings:
    :param model_name:
    :return: fig
    """
    # Predict  results
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Decode class labels
    classes = np.unique(y_test)
    class_labels = [label_mappings['class'][cls] for cls in classes]

    # Create a figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix for {model_name}')

    return fig


def plot_feature_importance(model: object, feature_names: object) -> object:
    """
    Function to create a feature importance plot.
    :rtype: object
    :param model:
    :param feature_names:
    :return: fig
    """

    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # descending order

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(range(len(importances)), importances[indices], align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance")

    fig.tight_layout()

    return fig


def plot_precision_recall_curve(model: object, X_test: object, y_test: object) -> object:
    """
    Function to create a precision-recall curve.
    :param model
    :param X_test
    :param y_test
    :return fig
    """

    from matplotlib.figure import Figure
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # geet predicted probabilities
    y_scores = model.predict_proba(X_test)

    # Compute precision-recall curve for each class
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for i in range(y_test_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        avg_precision = average_precision_score(y_test_bin[:, i], y_scores[:, i])
        ax.plot(recall, precision, label=f'Class {classes[i]} (AP = {avg_precision:.2f})')

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    fig.tight_layout()

    return fig


def plot_prediction_distribution(y_test: object, y_pred: object, label_mappings: object) -> object:
    """
        Function to create a distribution plot of predicted classes.
        :param y_test
        :param y_pred
        :param label_mappings
        :return fig
    """
    from matplotlib.figure import Figure
    import seaborn as sns

    # Decode numeric labels back to original class labels
    y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, 'class')
    y_pred_decoded = decode_predictions(pd.Series(y_pred), label_mappings, 'class')

    # Create a DataFrame for visualization
    df = pd.DataFrame({'True Class': y_test_decoded, 'Predicted Class': y_pred_decoded})

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    sns.countplot(data=df, x='Predicted Class', hue='True Class', ax=ax)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predictions")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    fig.tight_layout()

    return fig