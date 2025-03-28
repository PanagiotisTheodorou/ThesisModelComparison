from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from colorama import Fore, Style
from __preprocessing_utils__ import balance_dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_linR(df, target_column):
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
    print(Fore.GREEN + "\nTraining Linear Regression model" + Style.RESET_ALL)

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
        x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Train set shape: {x_train.shape}, Test set shape: {X_test.shape}")

        # Feature Scaling
        print("Scaling the features")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        X_test = scaler.transform(X_test)
        print("Features scaled successfully.")

        # Train the Linear Regression model
        print("Training the Linear Regression model")
        model = LinearRegression()
        model.fit(x_train, y_train)
        print("Model training completed.")

        #Make predictions
        print("Making predictions")
        predictions = model.predict(X_test)
        print("Predictions generated.")

        # Calculate metrics
        print("Calculating metrics")
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(Fore.LIGHTGREEN_EX + f"Mean Squared Error: {mse:.4f}" + Style.RESET_ALL)
        print(Fore.LIGHTGREEN_EX + f"R2 Score: {r2:.4f}" + Style.RESET_ALL)

        return model, x_train, X_test, y_train, y_test, mse, r2

    except Exception as e:
        print(Fore.RED + f"\nError during model training: {str(e)}" + Style.RESET_ALL)
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None
