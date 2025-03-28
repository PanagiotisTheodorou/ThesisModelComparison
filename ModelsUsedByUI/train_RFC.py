from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from colorama import Fore, Style

from __general_utils__ import decode_predictions

from __preprocessing_utils__ import balance_dataset, feature_selection

def train_RFC(df, target_column, label_mappings):
    print(Fore.GREEN + "\nTraining Random Forest model" + Style.RESET_ALL)

    try:
        # Balance the dataset
        print(Fore.LIGHTGREEN_EX + "Applying dataset balancing" + Style.RESET_ALL)

        df = balance_dataset(df, target_column)

        print(f"Balanced dataset shape: {df.shape}")

        # Split into features - X and target - y
        print("Splitting into features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Split into train and test sets
        print(Fore.LIGHTGREEN_EX + "Splitting Dataset" + Style.RESET_ALL)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Train set shape: {x_train.shape}, Test set shape: {x_test.shape}")

        # Hyperparameter tuning using GridSearchCV
        print(Fore.LIGHTGREEN_EX + "Hyperparameter tuning using GridSearchCV" + Style.RESET_ALL)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]
        }

        # Train the Random Forest Model
        print("Training the Random Forest Model")
        rf = RandomForestClassifier(class_weight="balanced", random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print("Model training completed.")

        # Print best parameters
        print(Fore.LIGHTGREEN_EX + f"Best Parameters: {grid_search.best_params_}" + Style.RESET_ALL)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions
        print("Making predictions")
        predictions = best_model.predict(x_test)
        print("Predictions generated.")

        # print accuracy
        accuracy_scr = accuracy_score(y_test, predictions)

        print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_score(y_test, predictions):.4f}" + Style.RESET_ALL)

        # Decode numeric labels back to original class labels
        y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
        predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)

        # Print classification report with decoded labels
        print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
        cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
        print(classification_report(y_test_decoded, predictions_decoded))

        return best_model, x_train, x_test, y_train, y_test, accuracy_scr, grid_search.best_params_, cr_dict

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        print(Fore.RED + "Traceback:" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None, None