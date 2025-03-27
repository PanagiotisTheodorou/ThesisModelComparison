
"""
    The following code trains a Random Forest Model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, \
    roc_auc_score
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from colorama import Fore, Style, init
import joblib
import warnings
from utils import load_data, remove_outliers, remove_unwanted_columns, fill_missing_values, encode_categorical

warnings.filterwarnings("ignore", category=FutureWarning)

#Initialize colorama
init(autoreset=True)

def balance_dataset(df, target_column):
    """
        Function to balance dataset
    """
    print(Fore.GREEN + "\nBalancing dataset by oversampling minority classes proportionally" + Style.RESET_ALL)

    # Count occurrences of each class
    class_counts = df[target_column].value_counts()
    majority_class = class_counts.idxmax()
    minority_classes = class_counts[class_counts.index != majority_class].index

    # Compute total occurrences of all minority classes
    total_minority_occurrences = class_counts[minority_classes].sum()
    num_minority_classes = len(minority_classes)

    # Determine target occurrences for each minority class
    target_counts = total_minority_occurrences // num_minority_classes
    sampling_strategy = {}

    for cls in minority_classes:
        if class_counts[cls] < target_counts:
            sampling_strategy[cls] = target_counts

    # Apply SMOTE for oversampling
    x = df.drop(columns=[target_column])
    y = df[target_column]
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    x_resampled, y_resampled = smote.fit_resample(x, y)
    resampled_df = pd.DataFrame(x_resampled, columns=x.columns)
    resampled_df[target_column] = y_resampled

    # Apply equal sampling of majority class instances
    final_data = []
    for cls in minority_classes:
        minority_samples = resampled_df[resampled_df[target_column] == cls]
        majority_samples = resampled_df[resampled_df[target_column] == majority_class].sample(n=len(minority_samples),
                                                                                              random_state=42)
        final_data.append(minority_samples)
        final_data.append(majority_samples)

    balanced_df = pd.concat(final_data).sample(frac=1, random_state=42).reset_index(drop=True)

    print(Fore.LIGHTGREEN_EX + "Dataset balanced.\n" + Style.RESET_ALL)

    return balanced_df


def train_model_random_forest(df, target_column, label_mappings, label_encoders):
    """
            Function to train the model, Steps:
            1. Filter rare classes, and for those apply the balancing logic
            2. split between dependent and non-dependent column
            3. Split the dataset to test and train
            4. Create a parameter grid for hyperparameter tuning
            5. Create and Train the model
            6. Apply Grid search so that the interpreter will loop through the available parameters and find the best ones
            7. Print out the best model, and the statistics, then return the chosen model
    """

    print(Fore.GREEN + "\nTrain model" + Style.RESET_ALL)

    # Balance the dataset
    print(Fore.LIGHTGREEN_EX + "Applying dataset balancing" + Style.RESET_ALL)

    df = balance_dataset(df, target_column)

    # Split into features - X and target - y
    x = df.drop(columns=[target_column])
    y = df[target_column]

    print(Fore.LIGHTGREEN_EX + "Splitting Dataset" + Style.RESET_ALL)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # Hyperparameter tuning using GridSearchCV
    print(Fore.LIGHTGREEN_EX + "Hyperparameter tuning using GridSearchCV" + Style.RESET_ALL)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Print best parameters
    print(Fore.LIGHTGREEN_EX + f"Best Parameters: {grid_search.best_params_}" + Style.RESET_ALL)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions
    predictions = best_model.predict(x_test)

    # print accuracy
    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_score(y_test, predictions):.4f}" + Style.RESET_ALL)

    # Decode numeric labels back to original class labels
    y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
    predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)

    # Print classification report with decoded labels
    print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
    print(classification_report(y_test_decoded, predictions_decoded))

    # Save the trained model and preprocessing objects
    model_data = {
        'model': best_model,
        'label_encoders': label_encoders,
        'label_mappings': label_mappings,
        'scaler': None
    }

    # Save pickle
    joblib.dump(model_data, '../005_UserUI/trained_random_forest_model.pkl')
    print(Fore.LIGHTGREEN_EX + "Model and preprocessing objects saved to 'trained_random_forest_model.pkl'." + Style.RESET_ALL)

    return best_model, x_train, x_test, y_train, y_test


def decode_predictions(predictions, label_mappings, column_name):
    """
        Convert numerical predictions back to categorical labels.
    """
    return predictions.map(label_mappings[column_name])


def check_overfitting(model, x_train, y_train, x_test, y_test):
    """
        Function to check for overfitting by comparing training and test accuracy.
        Also performs cross-validation to verify model generalization.
    """
    print(Fore.GREEN + "\nChecking for Overfitting" + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Compute and print accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training Accuracy: {train_acc:.4f}" + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX + f"Test Accuracy: {test_acc:.4f}" + Style.RESET_ALL)

    # Perform cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="accuracy")

    print(Fore.LIGHTGREEN_EX + f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}" + Style.RESET_ALL)

    # Check for overfitting
    if train_acc > test_acc + 0.05:  # If train accuracy is much higher than test accuracy
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)


def print_roc_auc(model, x_test, y_test, label_mappings, target_column):
    """
        Function to calculate and print AUC-ROC values for each class in the terminal.
        Uses class names instead of numeric labels.
    """
    # Binarize the labels for multi-class ROC-AUC calculation
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = model.predict_proba(x_test)  # Gets te probability scores

    print(Fore.GREEN + "\nAUC-ROC Values for Each Class:" + Style.RESET_ALL)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        # Decode numeric class to original class name
        class_name = label_mappings[target_column].get(cls, f"Class {cls}")
        print(Fore.CYAN + f"{class_name}: AUC = {roc_auc:.4f}" + Style.RESET_ALL)


def construct_confussion_matrix(model, x_test, y_test, label_mappings, model_name):
    """
        Function to print evaluation metrics for the model.
        It prints accuracy, weighted F1 score, confusion matrix,
        classification report, and multi-class ROC AUC score.
    """
    # Predict the results
    y_pred = model.predict(x_test)

    # Compute accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Compute weighted F1 score for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Compute ROC AUC using one-vs-rest strategy
    try:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_scores = model.predict_proba(x_test)
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

    # Calculate tp, tn, fp, fn for each class
    print(Fore.LIGHTGREEN_EX + "\nShowing the tp, tn, fp, fn rate for each class:" + Style.RESET_ALL)
    print('-------------------------')
    for i, class_label in enumerate(class_labels):  # Use decoded class labels
        tp = cm[i, i]  # True Positives for the current class
        fp = cm[:, i].sum() - tp  # False Positives for the current class
        fn = cm[i, :].sum() - tp  # False Negatives for the current class
        tn = cm.sum() - (tp + fp + fn)  # True Negatives for the current class

        print(Fore.YELLOW + f"Class {class_label}:" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Positives (tp): {tp}" + Style.RESET_ALL)
        print(Fore.RED + f"False Positives (fp): {fp}" + Style.RESET_ALL)
        print(Fore.RED + f"False Negatives (fn): {fn}" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Negatives (tn): {tn}" + Style.RESET_ALL)
        print('-------------------------')

    return accuracy, f1, roc_auc, cm, report


def main():
    file_path = '../000_Data/raw_with_general_classes.csv'
    target_column = 'class'

    start_time = time.time()

    print(Fore.CYAN + "\nStarting script execution" + Style.RESET_ALL)

    df = load_data(file_path)
    df = remove_unwanted_columns(df)
    df = remove_outliers(df)
    df = fill_missing_values(df)
    df, label_encoders, label_mappings = encode_categorical(df)

    # Print categorical mappings
    print(Fore.GREEN + "\nCategorical Data Mappings:" + Style.RESET_ALL)
    for col, mapping in label_mappings.items():
        print(f"{col}: {mapping}")

    df.to_csv("../000_Data/cleaned_dataset_after_rf.csv", index=False)
    print(Fore.GREEN + "Cleaned dataset saved!\n" + Style.RESET_ALL)

    print("Starting model training")
    model, x_train, x_test, y_train, y_test = train_model_random_forest(df, target_column, label_mappings, label_encoders)

    # Check for overfitting
    check_overfitting(model, x_train, y_train, x_test, y_test)

    # Construct and display confusion matrix and additional metrics in the console
    construct_confussion_matrix(model, x_test, y_test, label_mappings, model_name="Random Forest")

    end_time = time.time()
    print(Fore.GREEN + f"\nScript execution finished! Total time: {end_time - start_time:.2f} seconds\n" + Style.RESET_ALL)

    # Plot AUC-ROC curve instead of confusion matrix
    print_roc_auc(model, x_test, y_test, label_mappings, target_column)


if __name__ == "__main__":
    main()
