
"""
The following code trains a Random Forest Model by applying the strategy mentioned in the report
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score, \
    roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from colorama import colorama_text, Fore, Style, init
from win32trace import flush

#Initialize colorama
init(autoreset=True)


def load_data(file_path):
    """
        Function to load the data from the csv that is provided in the main def
    """
    print(Fore.GREEN + "\nLoading dataset..." + Style.RESET_ALL)

    df = pd.read_csv(file_path, na_values='?')  # Replace '?' with NaN

    print(Fore.LIGHTGREEN_EX + "Dataset loaded successfully!\n" + Style.RESET_ALL)
    print(df.head())
    return df


def remove_unwanted_columns(df):
    """
        Function to remove all the mutualle exclusive columns (Dimentionality Reduction)
        Because if not removed when filling the missing data, it will lead to an unbalanced dataset
    """
    print(Fore.GREEN + "\nRemoving unwanted columns..." + Style.RESET_ALL)

    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()] # Remove all columns that have to do with measured -> Mutually Exclusives
    df.drop(columns=columns_to_drop, inplace=True)

    print(Fore.LIGHTGREEN_EX + f"Dropped columns: {columns_to_drop}\n" + Style.RESET_ALL)
    return df


def remove_outliers(df):
    """
        Function to remove the outliers all numeric columns, namely there are some values in age that reach the 65,000 mark
    """

    print(Fore.GREEN + "\nRemoving outliers..." + Style.RESET_ALL)

    numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]

    print(Fore.LIGHTGREEN_EX + f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped." + Style.RESET_ALL)
    return df_no_outliers


def fill_missing_values(df):
    """
        Function to fill all the missing values using the mean of set columnn (numeric)
    """

    print(Fore.GREEN + "\nFilling missing values..." + Style.RESET_ALL)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    print(Fore.LIGHTGREEN_EX + "Missing values filled.\n" + Style.RESET_ALL)

    return df


def encode_categorical(df):
    """
    Function to encode categorical data using label encoding.
    This is done so that the model can understand the categories.
    It also stores the mapping for decoding labels later.
    """

    print(Fore.GREEN + "\nEncoding categorical variables..." + Style.RESET_ALL)

    label_encoders = {}
    label_mappings = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(enumerate(le.classes_))  # Store mapping

    print(Fore.LIGHTGREEN_EX + "Categorical columns encoded.\n" + Style.RESET_ALL)

    return df, label_encoders, label_mappings


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

    print(Fore.LIGHTGREEN_EX + "Dataset balanced.\n" + Style.RESET_ALL)

    return balanced_df


def feature_selection(X_train, y_train, X_test, threshold=0.01):
    """
    Perform feature selection using Random Forest feature importance.
    Features with importance greater than the threshold are selected.
    """
    print(Fore.GREEN + "\nPerforming feature selection..." + Style.RESET_ALL)

    # Train a Random Forest model to get feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display feature importances
    print(Fore.CYAN + "\nFeature Importances:" + Style.RESET_ALL)
    print(feature_importance_df)

    # Select features with importance greater than the threshold
    selected_features = feature_importance_df[feature_importance_df["Importance"] > threshold]["Feature"].tolist()

    print(Fore.LIGHTGREEN_EX + f"\nSelected Features (Importance > {threshold}):" + Style.RESET_ALL)
    print(selected_features)

    # Filter the datasets to include only selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected, feature_importance_df


def train_model(df, target_column, label_mappings, label_encoders):
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

    print(Fore.GREEN + "\nTraining model..." + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX + "Filtering rare classes..." + Style.RESET_ALL)

    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    print(Fore.LIGHTGREEN_EX + "Applying dataset balancing..." + Style.RESET_ALL)

    df = balance_dataset(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(Fore.LIGHTGREEN_EX + "Performing stratified train-test split..." + Style.RESET_ALL)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Perform feature selection
    # TODO: I can uncomment and comment this in order to add feature selection
    #X_train, X_test, feature_importance_df = feature_selection(X_train, y_train, X_test, threshold=0.01)

    print(Fore.LIGHTGREEN_EX + "Hyperparameter tuning using GridSearchCV..." + Style.RESET_ALL)

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

    print(Fore.LIGHTGREEN_EX + f"Best Parameters: {grid_search.best_params_}" + Style.RESET_ALL)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_score(y_test, predictions):.4f}" + Style.RESET_ALL)
    #print("Classification Report:\n", classification_report(y_test, predictions))

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
        'scaler': None  # Add scaler if used during training
    }

    # Save to a file
    joblib.dump(model_data, 'RandomForestModel.pkl')
    print(Fore.LIGHTGREEN_EX + "Model and preprocessing objects saved to 'trained_model.pkl'." + Style.RESET_ALL)

    return best_model, X_train, X_test, y_train, y_test


def decode_predictions(predictions, label_mappings, column_name):
    """
    Convert numerical predictions back to categorical labels.
    """
    return predictions.map(label_mappings[column_name])


def check_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Function to check for overfitting by comparing training and test accuracy.
    Also performs cross-validation to verify model generalization.
    """
    print(Fore.GREEN + "\nChecking for Overfitting..." + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training Accuracy: {train_acc:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test Accuracy: {test_acc:.4f}" + Style.RESET_ALL)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    print(Fore.LIGHTGREEN_EX + f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}" + Style.RESET_ALL)

    # Check for overfitting
    if train_acc > test_acc + 0.05:  # If train accuracy is much higher than test accuracy
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)


def plot_roc_auc(model, X_test, y_test):
    """
    Function to create a plot for the AUC and ROC curves.
    It takes the labels (classes), gets the probability metrics,
    and calculates the AUC ROC curve for all.
    """
    classes = np.unique(y_test)  # Dynamically get unique classes
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = model.predict_proba(X_test)  # Get probability scores

    plt.figure(figsize=(10, 6))

    for i in range(y_test_bin.shape[1]):
        if np.sum(y_test_bin[:, i]) == 0:  # Skip classes with no positive samples
            print(f"Skipping class {classes[i]} (no positive samples)")
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend()
    plt.show()



def construct_confussion_matrix(model, X_test, y_test, label_mappings, model_name):
    """
    Function to print evaluation metrics for the model.
    It prints accuracy, weighted F1 score, confusion matrix,
    classification report, and multi-class ROC AUC score.
    """
    # Predict the results
    y_pred = model.predict(X_test)

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
        y_scores = model.predict_proba(X_test)
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

    # Calculate TP, TN, FP, FN for each class
    print(Fore.LIGHTGREEN_EX + "\nShowing the TP, TN, FP, FN rate for each class:" + Style.RESET_ALL)
    print('-------------------------')
    for i, class_label in enumerate(class_labels):  # Use decoded class labels
        TP = cm[i, i]  # True Positives for the current class
        FP = cm[:, i].sum() - TP  # False Positives for the current class
        FN = cm[i, :].sum() - TP  # False Negatives for the current class
        TN = cm.sum() - (TP + FP + FN)  # True Negatives for the current class

        print(Fore.YELLOW + f"Class {class_label}:" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Positives (TP): {TP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Positives (FP): {FP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Negatives (FN): {FN}" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Negatives (TN): {TN}" + Style.RESET_ALL)
        print('-------------------------')

    return accuracy, f1, roc_auc, cm, report


def print_class_mapping():
    """
    Print the mapping of class labels to their corresponding diagnoses.
    """
    print(Fore.CYAN + "\nClass Label Mapping:" + Style.RESET_ALL)
    print(Fore.YELLOW + "\tLetter\tDiagnosis" + Style.RESET_ALL)
    print(Fore.YELLOW + "\t------\t---------" + Style.RESET_ALL)

    print(Fore.GREEN + "\nHyperthyroid Conditions:" + Style.RESET_ALL)
    print("\tA\thyperthyroid")
    print("\tB\tT3 toxic")
    print("\tC\ttoxic goitre")
    print("\tD\tsecondary toxic")

    print(Fore.GREEN + "\nHypothyroid Conditions:" + Style.RESET_ALL)
    print("\tE\thypothyroid")
    print("\tF\tprimary hypothyroid")
    print("\tG\tcompensated hypothyroid")
    print("\tH\tsecondary hypothyroid")

    print(Fore.GREEN + "\nBinding Protein:" + Style.RESET_ALL)
    print("\tI\tincreased binding protein")
    print("\tJ\tdecreased binding protein")

    print(Fore.GREEN + "\nGeneral Health:" + Style.RESET_ALL)
    print("\tK\tconcurrent non-thyroidal illness")

    print(Fore.GREEN + "\nReplacement Therapy:" + Style.RESET_ALL)
    print("\tL\tconsistent with replacement therapy")
    print("\tM\tunderreplaced")
    print("\tN\toverreplaced")

    print(Fore.GREEN + "\nAntithyroid Treatment:" + Style.RESET_ALL)
    print("\tO\tantithyroid drugs")
    print("\tP\tI131 treatment")
    print("\tQ\tsurgery")

    print(Fore.GREEN + "\nMiscellaneous:" + Style.RESET_ALL)
    print("\tR\tdiscordant assay results")
    print("\tS\televated TBG")
    print("\tT\televated thyroid hormones")



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

    df.to_csv("cleaned_dataset2.csv", index=False)
    print(Fore.GREEN + "Cleaned dataset saved!\n" + Style.RESET_ALL)

    print("Starting model training")
    model, X_train, X_test, y_train, y_test = train_model(df, target_column, label_mappings, label_encoders)

    # Check for overfitting
    check_overfitting(model, X_train, y_train, X_test, y_test)

    # Construct and display confusion matrix and additional metrics in the console
    construct_confussion_matrix(model, X_test, y_test, label_mappings, model_name="Random Forest")

    # Print class label mapping
    print_class_mapping()

    end_time = time.time()
    print(Fore.GREEN + f"\nScript execution finished! Total time: {end_time - start_time:.2f} seconds\n" + Style.RESET_ALL)

    # Plot AUC-ROC curve instead of confusion matrix
    plot_roc_auc(model, X_test, y_test)


if __name__ == "__main__":
    main()
