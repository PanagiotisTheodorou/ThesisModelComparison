
"""
The following code trains a Random Forest Model by applying the strategy mentioned in the report
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def load_data(file_path):
    """
        Function to load the 000_Data from the csv that is provided in the main def
    """
    df = pd.read_csv(file_path, na_values='?')  # Replace '?' with NaN
    print("Dataset loaded successfully!\n")
    print(df.head())
    return df


def remove_unwanted_columns(df):
    """
        Function to remove all the mutualle exclusive columns (Dimentionality Reduction)
        Because if not removed when filling the missing 000_Data, it will lead to an unbalanced dataset
    """
    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()] # Remove all columns that have to do with measured -> Mutually Exclusives
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns: {columns_to_drop}\n")
    return df


def remove_outliers(df):
    """
        Function to remove the outliers all numeric columns, namely there are some values in age that reach the 65,000 mark
    """
    numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]
    print(f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped.")
    return df_no_outliers


def fill_missing_values(df):
    """
        Function to fill all the missing values using the mean of set columnn (numeric)
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    print("Missing values filled.\n")
    return df


def encode_categorical(df):
    """
        Function to encode the categorical 000_Data using label encoding
        This is done so that the model can understand the categories
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Categorical columns encoded.\n")
    return df


def balance_dataset(df, target_column):
    """
        Function to balance dataset
        TODO add explanation for the formula
    """
    print("Balancing dataset by oversampling minority classes proportionally.")

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
    X = df.drop(columns=[target_column])
    y = df[target_column]
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
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
    print("Final class distribution after balancing:\n", balanced_df[target_column].value_counts())
    return balanced_df


def train_model(df, target_column):
    """
            Function to train the model, Steps:
            1. Filter rare classes, and for those apply the balancing logic
            2. split between dependent and non dependent column
            3. Split the dataset to test and train
            4. Create a parameter grid for hyperparameter tuning
            5. Create and Train the model
            6. Apply Grid search so that the interpreter will loop throygh the available parameters and find the best ones
            7. Print out the best model, and the statistics, then return the chosen model
        """
    print("Filtering rare classes")
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    print("Applying dataset balancing")
    df = balance_dataset(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Performing stratified train-test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Hyperparameter tuning using GridSearchCV")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    print("Model Evaluation\n")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    return best_model, X_train, X_test, y_train, y_test


def check_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Function to check for overfitting by comparing training and test accuracy.
    Also performs cross-validation to verify model generalization.
    """
    print("\nChecking for Overfitting")

    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Check for overfitting
    if train_acc > test_acc + 0.05:  # If train accuracy is much higher than test accuracy
        print("Possible Overfitting Detected!")
    else:
        print("No significant overfitting detected.")


def plot_roc_auc(model, X_test, y_test):
    """
        Function to create a plot for the AUC and ROC curves,
        It takes the labels (classes), and gets the propability metrics
        Then it calculates the AUC ROC curve for all
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 7, 8, 9, 11, 13, 17, 18])
    y_scores = model.predict_proba(X_test)  # Get probability scores

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



def main():
    file_path = 'raw.csv'
    target_column = 'class'

    start_time = time.time()
    print("Loading dataset")
    df = load_data(file_path)

    print("Removing unwanted columns")
    df = remove_unwanted_columns(df)

    print("Removing outliers")
    df = remove_outliers(df)

    print("Filling missing values")
    df = fill_missing_values(df)

    print("Encoding categorical variables")
    df = encode_categorical(df)

    df.to_csv("cleaned_dataset.csv", index=False)
    print("Cleaned dataset saved!\n")

    print("Starting model training")
    model, X_train, X_test, y_train, y_test = train_model(df, target_column)

    # Check for overfitting
    check_overfitting(model, X_train, y_train, X_test, y_test)

    # Plot AUC-ROC curve instead of confusion matrix
    plot_roc_auc(model, X_test, y_test)

    end_time = time.time()
    print(f"Script execution finished! Total time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
