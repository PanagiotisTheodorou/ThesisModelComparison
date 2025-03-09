import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    df = pd.read_csv(file_path, na_values='?')  # Replace '?' with NaN
    print("Dataset loaded successfully!\n")
    print(df.head())
    return df


def remove_unwanted_columns(df):
    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns: {columns_to_drop}\n")
    return df


def remove_outliers(df):
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
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    print("Missing values filled.\n")
    return df


def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Categorical columns encoded.\n")
    return df

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


def balance_data(X, y):
    print("Applying hybrid sampling (undersampling + oversampling)")

    # Get class distribution
    class_counts = pd.Series(y).value_counts()

    # Define undersampling strategy: reduce majority class to a certain number
    min_class_count = class_counts.min()  # Find the smallest class count
    undersample_strategy = {cls: min(min_class_count * 2, count) for cls, count in
                            class_counts.items()}  # Reduce majority

    undersampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
    X_under, y_under = undersampler.fit_resample(X, y)

    # Define oversampling strategy: balance all classes
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_balanced, y_balanced = oversampler.fit_resample(X_under, y_under)

    print(f"Class distribution after balancing:\n{pd.Series(y_balanced).value_counts()}\n")
    return X_balanced, y_balanced


from sklearn.model_selection import GridSearchCV

def train_model(df, target_column):
    print("Filtering rare classes")
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Performing stratified train-test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Applying hybrid sampling")
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    print("Hyperparameter tuning using GridSearchCV")
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)

    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)

    print("Model Evaluation\n")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))

    print("Plotting Confusion Matrix")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=valid_classes, yticklabels=valid_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()




def main():
    file_path = 'raw.csv'  # Update with actual dataset path
    target_column = 'class'  # Update with actual target column

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

    print("Starting Decision Tree model training")
    train_model(df, target_column)

    end_time = time.time()
    print(f"Script execution finished! Total time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
