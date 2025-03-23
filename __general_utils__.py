import pandas as pd
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Style

def decode_predictions(predictions: object, label_mappings: object, column_name: object) -> object:
    """
    Convert numerical predictions back to categorical labels.
    :rtype: object
    :param predictions:
    :param label_mappings:
    :param column_name:
    :return: predictions.map(label_mappings[column_name])
    """
    return predictions.map(label_mappings[column_name])

def remove_unwanted_columns(df: object) -> object:
    """
    Removes the unwanted columns
    :param df
    :return df
    :return columns_to_drop
    :rtype: object
    """
    print(Fore.GREEN + "\nRemoving unwanted columns" + Style.RESET_ALL)
    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()]
    df.drop(columns=columns_to_drop, inplace=True)
    print(Fore.LIGHTGREEN_EX + f"Dropped columns: {columns_to_drop}\n" + Style.RESET_ALL)
    return df, columns_to_drop


def remove_outliers(df: pd.DataFrame) -> object:
    """
    Removes the outliers
    :rtype: object
    :param df:
    :return: df
    :return df_no_outliers
    """
    print(Fore.GREEN + "\nRemoving outliers..." + Style.RESET_ALL)
    numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    q1 = df[numerical_columns].quantile(0.25)
    q3 = df[numerical_columns].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]
    print(Fore.LIGHTGREEN_EX + f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped." + Style.RESET_ALL)
    return df, df_no_outliers


def fill_missing_values(df: object) -> object:
    """
    Fills Missing Values
    :rtype: object
    :param df:
    :return: df
    """
    print(Fore.GREEN + "\nFilling missing values..." + Style.RESET_ALL)
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].mean())
    print(Fore.LIGHTGREEN_EX + "Missing values filled.\n" + Style.RESET_ALL)
    return df


def encode_categorical(df: object) -> object:
    """
    Encodes Categorical Data
    :rtype: object
    :param df:
    :return:
    """
    print(Fore.GREEN + "\nEncoding categorical variables..." + Style.RESET_ALL)
    label_encoders = {}
    label_mappings = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(enumerate(le.classes_))
    print(Fore.LIGHTGREEN_EX + "Categorical columns encoded.\n" + Style.RESET_ALL)
    return df, label_encoders, label_mappings
