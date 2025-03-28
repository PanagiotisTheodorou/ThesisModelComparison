import pandas as pd
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Style

def load_data(file_path):
    """
        Function to load the data from the csv that is provided in the main def
    """
    print(Fore.GREEN + "\nLoading dataset" + Style.RESET_ALL)

    df = pd.read_csv(file_path, na_values='?')  # Replace '?' with NaN

    print(Fore.LIGHTGREEN_EX + "Dataset loaded successfully!\n" + Style.RESET_ALL)
    print(df.head())
    return df

def remove_unwanted_columns(df):
    """
        Function to remove all the mutualle exclusive columns (Dimentionality Reduction)
        Because if not removed when filling the missing data, it will lead to an unbalanced dataset
    """
    print(Fore.GREEN + "\nRemoving unwanted columns" + Style.RESET_ALL)

    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()] # Remove all columns that have to do with measured -> Mutually Exclusives
    df.drop(columns=columns_to_drop, inplace=True)

    print(Fore.LIGHTGREEN_EX + f"Dropped columns: {columns_to_drop}\n" + Style.RESET_ALL)
    return df

def remove_outliers(df):
    """
        Function to remove the outliers all numeric columns, namely there are some values in age that reach the 65,000 mark
    """

    print(Fore.GREEN + "\nRemoving outliers" + Style.RESET_ALL)

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

    print(Fore.GREEN + "\nFilling missing values" + Style.RESET_ALL)

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

    print(Fore.GREEN + "\nEncoding categorical variables" + Style.RESET_ALL)

    label_encoders = {}
    label_mappings = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(enumerate(le.classes_))  # Store mapping

    print(Fore.LIGHTGREEN_EX + "Categorical columns encoded.\n" + Style.RESET_ALL)

    return df, label_encoders, label_mappings