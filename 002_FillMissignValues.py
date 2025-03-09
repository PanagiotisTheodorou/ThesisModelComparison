import pandas as pd

# Load the dataset
df = pd.read_csv('ThyroxineDataLastDataset.csv')

# Replace '?' with NaN to properly identify missing values
df.replace('?', pd.NA, inplace=True)

# Convert numerical columns to numeric type (coerce invalid values to NaN)
numeric_columns = ["TSH", "T3", "TT4", "T4U", "FTI"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# List of measurement columns and their corresponding value columns
measurement_value_pairs = {
    "TSH measured": "TSH",
    "T3 measured": "T3",
    "TT4 measured": "TT4",
    "T4U measured": "T4U",
    "FTI measured": "FTI"
}

# Fill missing values based on measurement columns
for measure_col, value_col in measurement_value_pairs.items():
    df.loc[df[measure_col] == False, value_col] = 0  # If measurement is False, set value to 0
    df[value_col] = df[value_col].fillna(df[value_col].mean())  # Fill remaining NaNs with mean

# Fill missing categorical values with the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Ensure all numeric values have the same format (float with two decimals)
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.round(2))

# Save the cleaned dataset
df.to_csv('ThyroxineData_Cleaned_v3.csv', index=False)

print("Missing values handled and dataset saved as 'ThyroxineData_Cleaned_v2.csv'")
