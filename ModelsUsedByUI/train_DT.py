from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import pandas as pd
from colorama import Fore, Style
from __general_utils__ import decode_predictions
from __preprocessing_utils__ import balance_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_DT(df, target_column, label_mappings):

    print(Fore.GREEN + "\nTraining Decision Tree model" + Style.RESET_ALL)

    try:
        # Balance the dataset
        print(Fore.LIGHTGREEN_EX + "Applying dataset balancing" + Style.RESET_ALL)

        df = balance_dataset(df, target_column)

        print(f"Balanced dataset shape: {df.shape}")

        # Split into features and target
        print("Splitting into features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Split into train and test sets
        print("Splitting into train and test sets")
        x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train set shape: {x_train.shape}, Test set shape: {X_test.shape}")

        # Train the Decision Tree model
        print("Training the Decision Tree model")
        model = DecisionTreeClassifier(random_state=42, criterion="gini", max_depth=10)
        model.fit(x_train, y_train)
        print("Model training completed.")

        #Make predictions
        print("Making predictions")
        predictions = model.predict(X_test)
        print("Predictions generated.")

        #Calculate metrics
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

        return model, x_train, X_test, y_train, y_test, accuracy_scr, {}, cr_dict

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + "Traceback:" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None
