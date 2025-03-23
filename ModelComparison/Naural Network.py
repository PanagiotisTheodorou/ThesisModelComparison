import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize, StandardScaler
from colorama import Fore, Style, init
import joblib
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize colorama
init(autoreset=True)

from utils import load_data, remove_outliers, remove_unwanted_columns, fill_missing_values, encode_categorical

def balance_dataset(df, target_column):
    """
    Function to balance dataset
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

def train_model_nn(df, target_column, label_mappings, label_encoders):
    """
    Function to train a Neural Network model.
    """
    print(Fore.GREEN + "\nTraining Neural Network model..." + Style.RESET_ALL)

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

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define the neural network model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y_train_encoded)), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, y_train_encoded, epochs=100, batch_size=32,
                        validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Make predictions
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Print accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy:.4f}" + Style.RESET_ALL)

    # Decode numeric labels back to original class labels
    y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
    predictions_decoded = label_encoder.inverse_transform(y_pred)

    # Print classification report with decoded labels
    print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
    print(classification_report(y_test_decoded, predictions_decoded))

    # Save the trained model and preprocessing objects
    model_data = {
        'model': model,
        'label_encoders': label_encoders,
        'label_mappings': label_mappings,
        'scaler': scaler  # Save the scaler for later use
    }

    # Save to a file
    joblib.dump(model_data, '../trained_nn_model.pkl')
    print(Fore.LIGHTGREEN_EX + "Model and preprocessing objects saved to 'trained_nn_model.pkl'." + Style.RESET_ALL)

    return model, X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

def check_overfitting(model, x_train, y_train, x_test, y_test):
    """
    Function to check for overfitting by comparing training and test accuracy.
    """
    print(Fore.GREEN + "\nChecking for Overfitting..." + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = np.argmax(model.predict(x_train), axis=1)
    y_test_pred = np.argmax(model.predict(x_test), axis=1)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training Accuracy: {train_acc:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test Accuracy: {test_acc:.4f}" + Style.RESET_ALL)

    # Check for overfitting
    if train_acc > test_acc + 0.05:  # If train accuracy is much higher than test accuracy
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)

def plot_roc_auc(model, x_test, y_test):
    """
    Function to create a plot for the AUC and ROC curves.
    """
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_scores = model.predict(x_test)  # Get probability scores

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
    y_scores = model.predict(x_test)  # Get probability scores

    print(Fore.GREEN + "\nAUC-ROC Values for Each Class:" + Style.RESET_ALL)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        # Decode numeric class to original class name
        class_name = label_mappings[target_column].get(cls, f"Class {cls}")
        print(Fore.CYAN + f"{class_name}: AUC = {roc_auc:.4f}" + Style.RESET_ALL)

def construct_confussion_matrix(model, x_test, y_test, label_mappings, model_name):
    """
    Function to print evaluation metrics for the model.
    """
    # Predict the results
    y_pred = np.argmax(model.predict(x_test), axis=1)

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
        y_scores = model.predict(x_test)
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

    # Calculate tp, tn, fp, fn for each class
    print(Fore.LIGHTGREEN_EX + "\nShowing the tp, tn, fp, fn rate for each class:" + Style.RESET_ALL)
    print('-------------------------')
    for i, class_label in enumerate(class_labels):  # Use decoded class labels
        tp = cm[i, i]  # True Positives for the current class
        fp = cm[:, i].sum() - tp  # False Positives for the current class
        fn = cm[i, :].sum() - tp  # False Negatives for the current class
        tn = cm.sum() - (tp + fp + fn)  # True Negatives for the current class

        print(Fore.YELLOW + f"Class {class_label}:" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Positives (tp): {tp}" + Style.RESET_ALL)
        print(Fore.RED + f"False Positives (fp): {fp}" + Style.RESET_ALL)
        print(Fore.RED + f"False Negatives (fn): {fn}" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Negatives (tn): {tn}" + Style.RESET_ALL)
        print('-------------------------')

    return accuracy, f1, roc_auc, cm, report

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

    print("Starting Neural Network model training")
    model, X_train_scaled, X_test_scaled, y_train, y_test = train_model_nn(df, target_column, label_mappings, label_encoders)

    # Check for overfitting
    check_overfitting(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Construct and display confusion matrix and additional metrics in the console
    construct_confussion_matrix(model, X_test_scaled, y_test, label_mappings, model_name="Neural Network")

    end_time = time.time()
    print(Fore.GREEN + f"\nScript execution finished! Total time: {end_time - start_time:.2f} seconds\n" + Style.RESET_ALL)

    # Plot AUC-ROC curve instead of confusion matrix
    print_roc_auc(model, X_test_scaled, y_test, label_mappings, target_column)

if __name__ == "__main__":
    main()