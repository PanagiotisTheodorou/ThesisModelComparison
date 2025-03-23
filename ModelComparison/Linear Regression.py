
"""
The following code trains a Random Forest Model by applying the strategy mentioned in the report
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, \
    roc_auc_score
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from colorama import Fore, Style, init
import joblib
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

#Initialize colorama
init(autoreset=True)

from utils import load_data, remove_outliers, remove_unwanted_columns, fill_missing_values, encode_categorical


def balance_dataset(df, target_column):
    """
        Function to balance dataset
        TODO add explanation for the formula
    """
    print(Fore.GREEN + "\nBalancing dataset by oversampling minority classes proportionally..." + Style.RESET_ALL)

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


def feature_selection(x_train, y_train, x_test, threshold=0.01):
    """
    Perform feature selection using Random Forest feature importance.
    Features with importance greater than the threshold are selected.
    """
    print(Fore.GREEN + "\nPerforming feature selection..." + Style.RESET_ALL)

    # Train a Random Forest model to get feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)

    # Get feature importance
    importances = rf.feature_importances_
    feature_names = x_train.columns

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display feature importances
    print(Fore.CYAN + "\nFeature importances:" + Style.RESET_ALL)
    print(feature_importance_df)

    # Select features with importance greater than the threshold
    selected_features = feature_importance_df[feature_importance_df["Importance"] > threshold]["Feature"].tolist()

    print(Fore.LIGHTGREEN_EX + f"\nSelected Features (Importance > {threshold}):" + Style.RESET_ALL)
    print(selected_features)

    # Filter the datasets to include only selected features
    x_train_selected = x_train[selected_features]
    x_test_selected = x_test[selected_features]

    return x_train_selected, x_test_selected, feature_importance_df

def train_model_linear_regression(df, target_column, label_mappings, label_encoders):
    """
    Function to train a Linear Regression model. Steps:
    1. Filter rare classes, and for those apply the balancing logic.
    2. Split between dependent and non-dependent columns.
    3. Split the dataset into test and train.
    4. Perform feature scaling.
    5. Create and train the model.
    6. Evaluate the model using RMSE and R2 score.
    7. Print out the model statistics, then return the chosen model.
    """
    print(Fore.GREEN + "\nTraining Linear Regression model..." + Style.RESET_ALL)

    # Filter rare classes
    print(Fore.LIGHTGREEN_EX + "Filtering rare classes..." + Style.RESET_ALL)
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    # Balance the dataset
    print(Fore.LIGHTGREEN_EX + "Applying dataset balancing..." + Style.RESET_ALL)
    df = balance_dataset(df, target_column)

    # Split into features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Perform stratified train-test split
    print(Fore.LIGHTGREEN_EX + "Performing stratified train-test split..." + Style.RESET_ALL)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Perform feature scaling
    print(Fore.LIGHTGREEN_EX + "Performing feature scaling..." + Style.RESET_ALL)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Linear Regression model
    print(Fore.LIGHTGREEN_EX + "Training Linear Regression model..." + Style.RESET_ALL)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Make predictions
    predictions = lr.predict(X_test_scaled)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(Fore.LIGHTGREEN_EX + f"Root Mean Squared Error (RMSE): {rmse:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"R2 Score: {r2:.4f}" + Style.RESET_ALL)

    # Save the trained model and preprocessing objects
    model_data = {
        'model': lr,
        'label_encoders': label_encoders,
        'label_mappings': label_mappings,
        'scaler': scaler  # Save the scaler for later use
    }

    # Save to a file
    joblib.dump(model_data, '../trained_linear_regression_model.pkl')
    print(Fore.LIGHTGREEN_EX + "Model and preprocessing objects saved to 'trained_linear_regression_model.pkl'." + Style.RESET_ALL)

    return lr, X_train_scaled, X_test_scaled, y_train, y_test


def decode_predictions(predictions, label_mappings, column_name):
    """
    Convert numerical predictions back to categorical labels.
    """
    return predictions.map(label_mappings[column_name])


def check_overfitting(model, x_train, y_train, x_test, y_test):
    """
    Function to check for overfitting by comparing training and test RMSE and R2 scores.
    Also performs cross-validation to verify model generalization.
    """
    print(Fore.GREEN + "\nChecking for Overfitting..." + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Compute RMSE and R2 scores for training and test sets
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training RMSE: {train_rmse:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test RMSE: {test_rmse:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Training R2 Score: {train_r2:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test R2 Score: {test_r2:.4f}" + Style.RESET_ALL)

    # Perform cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv_scores.mean())
    print(Fore.LIGHTGREEN_EX + f"Cross-Validation RMSE: {cv_rmse:.4f} +/- {np.sqrt(-cv_scores).std():.4f}" + Style.RESET_ALL)

    # Check for overfitting
    if train_rmse < test_rmse - 0.05:  # If training RMSE is much lower than test RMSE
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)


def plot_roc_auc(model, x_test, y_test):
    """
        Function to create a plot for the AUC and ROC curves,
        It takes the labels (classes), and gets the probability metrics
        Then it calculates the AUC ROC curve for all
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 7, 8, 9, 11, 13, 17, 18])
    y_scores = model.predict_proba(x_test)  # Get probability scores

    plt.figure(figsize=(10, 6))

    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend()
    plt.show()

def print_roc_auc(model, x_test, y_test, label_mappings, target_column):
    """
    Function to calculate and print AUC-ROC values for each class in the terminal.
    Uses class names instead of numeric labels.
    """
    # Binarize the labels for multi-class ROC-AUC calculation
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = model.predict_proba(x_test)  # Get probability scores

    print(Fore.GREEN + "\nAUC-ROC Values for Each Class:" + Style.RESET_ALL)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        # Decode numeric class to original class name
        class_name = label_mappings[target_column].get(cls, f"Class {cls}")
        print(Fore.CYAN + f"{class_name}: AUC = {roc_auc:.4f}" + Style.RESET_ALL)


def evaluate_regression_model(model, x_test, y_test, model_name):
    """
    Function to evaluate a regression model.
    It prints RMSE, R2 score, and other regression metrics.
    """
    # Predict the results
    y_pred = model.predict(x_test)

    # Compute RMSE and R2 score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(Fore.GREEN + f"\n--- {model_name} Evaluation ---\n" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Root Mean Squared Error (RMSE): {rmse:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"R2 Score: {r2:.4f}" + Style.RESET_ALL)

    return rmse, r2


def main():
    file_path = 'raw_with_general_classes.csv'
    target_column = 'class'

    start_time = time.time()

    print(Fore.CYAN + "\nStarting script execution..." + Style.RESET_ALL)

    df = load_data(file_path)
    df = remove_unwanted_columns(df)
    df = remove_outliers(df)
    df = fill_missing_values(df)
    df, label_encoders, label_mappings = encode_categorical(df)

    # Print categorical mappings
    print(Fore.GREEN + "\nCategorical Data Mappings:" + Style.RESET_ALL)
    for col, mapping in label_mappings.items():
        print(f"{col}: {mapping}")

    df.to_csv("cleaned_dataset.csv", index=False)
    print(Fore.GREEN + "Cleaned dataset saved!\n" + Style.RESET_ALL)

    print("Starting Linear Regression model training")
    model, X_train_scaled, X_test_scaled, y_train, y_test = train_model_linear_regression(df, target_column, label_mappings, label_encoders)

    # Check for overfitting
    check_overfitting(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Evaluate the regression model
    evaluate_regression_model(model, X_test_scaled, y_test, model_name="Linear Regression")

    end_time = time.time()
    print(Fore.GREEN + f"\nScript execution finished! Total time: {end_time - start_time:.2f} seconds\n" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
