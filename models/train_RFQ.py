from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from colorama import Fore, Style

from __general_utils__ import decode_predictions

from __preprocessing_utils__ import balance_dataset, feature_selection

# def train_model(df, target_column, label_mappings):
#     """
#             Function to train the model, Steps:
#             1. Filter rare classes, and for those apply the balancing logic
#             2. split between dependent and non dependent column
#             3. Split the dataset to test and train
#             4. Create a parameter grid for hyperparameter tuning
#             5. Create and Train the model
#             6. Apply Grid search so that the interpreter will loop throygh the available parameters and find the best ones
#             7. Print out the best model, and the statistics, then return the chosen model
#         """
#
#     print(Fore.GREEN + "\nTraining model..." + Style.RESET_ALL)
#
#     print(Fore.LIGHTGREEN_EX + "Filtering rare classes..." + Style.RESET_ALL)
#
#     class_counts = df[target_column].value_counts()
#     valid_classes = class_counts[class_counts >= 10].index
#     df = df[df[target_column].isin(valid_classes)]
#
#     print(Fore.LIGHTGREEN_EX + "Applying dataset balancing..." + Style.RESET_ALL)
#
#     df = balance_dataset(df, target_column)
#
#     X = df.drop(columns=[target_column])
#     y = df[target_column]
#
#     print(Fore.LIGHTGREEN_EX + "Performing stratified train-test split..." + Style.RESET_ALL)
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
#     # Perform feature selection
#     # TODO: I can uncomment and comment this in order to add feature selection
#     #X_train, X_test, feature_importance_df = feature_selection(X_train, y_train, X_test, threshold=0.01)
#
#     print(Fore.LIGHTGREEN_EX + "Hyperparameter tuning using GridSearchCV..." + Style.RESET_ALL)
#
#     param_grid = {
#         "n_estimators": [50, 100, 200],
#         "max_depth": [5, 10, 15, None],
#         "min_samples_split": [2, 5, 10],
#         "min_samples_leaf": [1, 2, 4],
#         "bootstrap": [True, False]
#     }
#
#     rf = RandomForestClassifier(class_weight="balanced", random_state=42)
#     grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
#     grid_search.fit(X_train, y_train)
#
#     print(Fore.LIGHTGREEN_EX + f"Best Parameters: {grid_search.best_params_}" + Style.RESET_ALL)
#
#     best_model = grid_search.best_estimator_
#
#     predictions = best_model.predict(X_test)
#
#     accuracy_scr = accuracy_score(y_test, predictions)
#
#     print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_score(y_test, predictions):.4f}" + Style.RESET_ALL)
#
#     # Decode numeric labels back to original class labels
#     y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
#     predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)
#
#     # Print classification report with decoded labels
#     print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
#     cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
#     print(classification_report(y_test_decoded, predictions_decoded))
#
#     return best_model, X_train, X_test, y_train, y_test, accuracy_scr, grid_search.best_params_, cr_dict

def train_model(df, target_column, label_mappings):
    """
    Function to train the model without hyperparameter tuning.
    Steps:
    1. Filter rare classes, and for those apply the balancing logic
    2. Split between dependent and independent columns
    3. Split the dataset into test and train
    4. Train the model with default or manually specified parameters
    5. Print out the model statistics and return the trained model
    """

    print(Fore.GREEN + "\nTraining model..." + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX + "Filtering rare classes..." + Style.RESET_ALL)

    # Filter rare classes
    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    print(Fore.LIGHTGREEN_EX + "Applying dataset balancing..." + Style.RESET_ALL)

    # Balance the dataset
    df = balance_dataset(df, target_column)

    # Split into features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(Fore.LIGHTGREEN_EX + "Performing stratified train-test split..." + Style.RESET_ALL)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(Fore.LIGHTGREEN_EX + "Training Random Forest model..." + Style.RESET_ALL)

    # Train the Random Forest model with default or manually specified parameters
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_estimators=100,  # Default is 100
        max_depth=None,     # Default is None (nodes are expanded until all leaves are pure)
        min_samples_split=2,  # Default is 2
        min_samples_leaf=1,   # Default is 1
        bootstrap=True        # Default is True
    )
    rf.fit(X_train, y_train)

    # Make predictions
    predictions = rf.predict(X_test)

    # Calculate accuracy
    accuracy_scr = accuracy_score(y_test, predictions)

    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_scr:.4f}" + Style.RESET_ALL)

    # Decode numeric labels back to original class labels
    y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
    predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)

    # Print classification report with decoded labels
    print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
    cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
    print(classification_report(y_test_decoded, predictions_decoded))

    return rf, X_train, X_test, y_train, y_test, accuracy_scr, {}, cr_dict