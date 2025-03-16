import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
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

def train_model(df, target_column, min_samples=10):  # Increase threshold
    print("Filtering rare classes")
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df = df[df[target_column].isin(valid_classes)]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print("Performing stratified train-test split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Handling class imbalance with SMOTE")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("Standardizing features")
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    print("Training Logistic Regression with class balancing")
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    predictions = model.predict(X_test)

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

    print("Starting model training")
    train_model(df, target_column)

    end_time = time.time()
    print(f"Script execution finished! Total time: {end_time - start_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
