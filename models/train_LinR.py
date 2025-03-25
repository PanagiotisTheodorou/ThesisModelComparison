from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from colorama import Fore, Style
from __preprocessing_utils__ import balance_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_linear_regression(df, target_column):
    """
    Function to train a Linear Regression model with error handling and debugging.
    """
    print(Fore.GREEN + "\nTraining Linear Regression model..." + Style.RESET_ALL)

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Step 5: Scale the features
        print("Step 5: Scaling the features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("Features scaled successfully.")

        # Step 6: Train the Linear Regression model
        print("Step 6: Training the Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("Model training completed.")

        # Step 7: Make predictions
        print("Step 7: Making predictions...")
        predictions = model.predict(X_test)
        print("Predictions generated.")

        # Step 8: Calculate metrics
        print("Step 8: Calculating metrics...")
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(Fore.LIGHTGREEN_EX + f"Mean Squared Error: {mse:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"R2 Score: {r2:.4f}" + Style.RESET_ALL)

        return model, X_train, X_test, y_train, y_test, mse, r2

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None
