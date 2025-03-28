import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from colorama import Fore, Style


def check_overfitting(model, X_train, y_train, X_test, y_test):
    """
        Function to check for overfitting by comparing training and test accuracy.
        Also performs cross-validation to verify model generalization.
    """
    print(Fore.GREEN + "\nChecking for Overfitting" + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training Accuracy: {train_acc:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test Accuracy: {test_acc:.4f}" + Style.RESET_ALL)

    # prform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    cross_acc = cv_scores.mean()
    cross_std = cv_scores.std()
    print(Fore.LIGHTGREEN_EX + f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}" + Style.RESET_ALL)

    #check for overfitting
    if train_acc > test_acc + 0.05:
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)

    return train_acc, test_acc, cross_acc, cross_std


def construct_confussion_matrix_logical(model, X_test, y_test, label_mappings, model_name):
    """
        Function to print evaluation metrics for the model.
        It prints accuracy, weighted F1 score, confusion matrix,
        classification report, and mlti-class ROC AUC score.
    """
    y_pred = model.predict(X_test)

    # Compute accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    #compute weighted F1 score for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')

    # compute ROC AUC using one-vs-rest strategy
    try:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_scores = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
    except Exception as e:
        roc_auc = "Not Available"

    # Print the evaluation metrics
    print(Fore.GREEN + f"\n--- {model_name} Evaluation ---\n" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy} | {accuracy:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Weighted F1 Score: {f1} | {f1:.4f}\n" + Style.RESET_ALL)

    # Decode class labels for confusion matrix
    class_labels = [label_mappings['class'][cls] for cls in classes]

    # Print labeled confusion matrix
    print(Fore.CYAN + "Confusion Matrix (with class labels):" + Style.RESET_ALL)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm_df)
    print("\n")

    print(Fore.LIGHTGREEN_EX + f"Multi-Class ROC AUC (One-vs-Rest): {roc_auc}" + Style.RESET_ALL)

    # Calculate TP, TN, FP, FN for each class
    print(Fore.LIGHTGREEN_EX + "\nShowing the TP, TN, FP, FN rate for each class:" + Style.RESET_ALL)
    print('-------------------------')
    for i, class_label in enumerate(class_labels):  # Usees decoded class labels
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        print(Fore.YELLOW + f"Class {class_label}:" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Positives (TP): {TP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Positives (FP): {FP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Negatives (FN): {FN}" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Negatives (TN): {TN}" + Style.RESET_ALL)
        print('-------------------------')

    return accuracy, f1, roc_auc, cm, report