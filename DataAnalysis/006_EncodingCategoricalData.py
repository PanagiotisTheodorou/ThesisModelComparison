import pandas as pd
import tkinter as tk
from tkinter import scrolledtext

"""
    The code performs the following:
        1. Loads the dataset
        2. Identifies the categorical columns, except diagnosis (previous script showed that it had many unique values)
        3. Applies One-Hot Encoding through a custom function
            for binary columns, it defines a single binary columns
            for multiple category columns, it defines a binary category for each category
        4. Saves the encoded dataset
        5. Creates a window to display the mapping
"""

# Load the dataset
file_path = "../data/ThyroxineData_Cleaned_NoOutliers.csv"
df = pd.read_csv(file_path)

# Identify categorical columns (excluding 'diagnosis')
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('diagnosis')  # Ensure the target column is not encoded


# Apply One-Hot Encoding (OHE) with specific binary mapping
def custom_ohe(df, categorical_columns):
    df_encoded = df.copy()
    mapping = {}

    for col in categorical_columns:
        unique_values = df[col].unique()
        if len(unique_values) == 2:  # Binary categorical column
            df_encoded[f'{col}_binary'] = (df[col] == unique_values[1]).astype(int)
            df_encoded.drop(columns=[col], inplace=True)
            mapping[col] = {unique_values[0]: '[0,0]', unique_values[1]: '[1,0]'}
        else:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(columns=[col], inplace=True)
            mapping[col] = {val: list(dummies.columns[dummies[df[col] == val].iloc[0] == 1]) for val in unique_values}

    return df_encoded, mapping


# Apply the custom OHE function
df_encoded, mapping = custom_ohe(df, categorical_columns)

# Save the transformed dataset to a new CSV file
output_file = "../data/ThyroxineData_Encoded.csv"
df_encoded.to_csv(output_file, index=False)

# Create a GUI window to display the mapping
root = tk.Tk()
root.title("Categorical Encoding Mapping")
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
text_area.pack(padx=10, pady=10)

text_area.insert(tk.END, "One-Hot Encoding Mapping:\n\n")
for col, values in mapping.items():
    text_area.insert(tk.END,
                     f"{col}: before had values {list(values.keys())}, now has values {list(values.values())}\n")

text_area.config(state=tk.DISABLED)
root.mainloop()