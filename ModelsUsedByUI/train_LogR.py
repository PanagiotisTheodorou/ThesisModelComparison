from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pandas as pd
from colorama import Fore, Style
from __general_utils__ import decode_predictions
from __preprocessing_utils__ import balance_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_LogR(df, target_column, label_mappings):
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
    print(Fore.GREEN + "\nTraining Logistic Regression model" + Style.RESET_ALL)

    try:
        # Balance the dataset
        print("Balancing the dataset")

        df = balance_dataset(df, target_column)

        print(f"Balanced dataset shape: {df.shape}")

        # Split into features - X and target - y
        print("Splitting into features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        #Split into train and test sets
        print("Splitting into train and test sets")
        X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train set shape: {X_train.shape}, Test set shape: {x_test.shape}")

        # Scale the features
        print("Scaling the features")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        x_test = scaler.transform(x_test)
        print("Features scaled successfully.")

        # Train the Logistic Regression model
        print("Training the Logistic Regression model")
        model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=5000)
        model.fit(X_train, y_train)
        print("Model training completed.")

        # Make predictions
        print("Making predictions")
        predictions = model.predict(x_test)
        print("Predictions generated.")

        # Calculate metrics
        print("Calculating metrics")
        accuracy_scr = accuracy_score(y_test, predictions)
        precision_scr = precision_score(y_test, predictions, average="weighted")
        recall_scr = recall_score(y_test, predictions, average="weighted")
        f1_scr = f1_score(y_test, predictions, average="weighted")

        print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"Precision: {precision_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"Recall: {recall_scr:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"F1-Score: {f1_scr:.4f}" + Style.RESET_ALL)

        # Decode numeric labels back to original class labels
        print("Decoding predictions")
        y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
        predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)
        print("Predictions decoded.")

        # Print classification report
        print("Generating classification report")
        cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
        print(classification_report(y_test_decoded, predictions_decoded))

        return model, X_train, x_test, y_train, y_test, accuracy_scr, {}, cr_dict

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + "Traceback:" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None