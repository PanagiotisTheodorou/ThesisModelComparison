from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pandas as pd
from colorama import Fore, Style
from general_utils import decode_predictions
from preprocessing_utils import balance_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_LF(df, target_column, label_mappings):
    """
    Function to train a Logistic Regression model with error handling and debugging.
    """
    print(Fore.GREEN + "\nTraining Logistic Regression model..." + Style.RESET_ALL)

    try:
        # Step 1: Filter rare classes
        print("Step 1: Filtering rare classes...")
        class_counts = df[target_column].value_counts()
        valid_classes = class_counts[class_counts >= 10].index
        df = df[df[target_column].isin(valid_classes)]
        print(f"Filtered dataset shape: {df.shape}")

        # Step 2: Balance the dataset
        print("Step 2: Balancing the dataset...")
        df = balance_dataset(df, target_column)
        print(f"Balanced dataset shape: {df.shape}")

        # Step 3: Split into features and target
        print("Step 3: Splitting into features and target...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Step 4: Split into train and test sets
        print("Step 4: Splitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Step 5: Scale the features
        print("Step 5: Scaling the features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Features scaled successfully.")

        # Step 6: Train the Logistic Regression model
        print("Step 6: Training the Logistic Regression model...")
        model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        print("Model training completed.")

        # Step 7: Make predictions
        print("Step 7: Making predictions...")
        predictions = model.predict(X_test)
        print("Predictions generated.")

        # Step 8: Calculate metrics
        print("Step 8: Calculating metrics...")
        accuracy_scr = accuracy_score(y_test, predictions)
        precision_scr = precision_score(y_test, predictions, average="weighted")
        recall_scr = recall_score(y_test, predictions, average="weighted")
        f1_scr = f1_score(y_test, predictions, average="weighted")

        print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"Precision: {precision_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"Recall: {recall_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"F1-Score: {f1_scr:.4f}" + Style.RESET_ALL)

        # Step 9: Decode numeric labels back to original class labels
        print("Step 9: Decoding predictions...")
        y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
        predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)
        print("Predictions decoded.")

        # Step 10: Print classification report
        print("Step 10: Generating classification report...")
        cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
        print(classification_report(y_test_decoded, predictions_decoded))

        return model, X_train, X_test, y_train, y_test, accuracy_scr, {}, cr_dict

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + "Traceback:" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None