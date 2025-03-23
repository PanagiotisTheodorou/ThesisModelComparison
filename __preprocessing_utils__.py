from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from colorama import Fore, Style


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