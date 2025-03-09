import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "ThyroxineData_Cleaned_NoOutliers.csv"
df = pd.read_csv(file_path)

# Identify categorical columns (non-numeric)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Store unique values for each categorical column before encoding
original_categories = {col: df[col].unique().tolist() for col in categorical_columns}

# Apply One-Hot Encoding (OHE)
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False)  # Keep all categories for reference

# Store new column names for encoded categories
encoded_categories = [col for col in df_encoded.columns if col not in df.columns or col in categorical_columns]

# Create a mapping table of original vs encoded values
encoding_map = []
for col in categorical_columns:
    for val in original_categories[col]:
        encoded_cols = [f"{col}_{val}" if f"{col}_{val}" in df_encoded.columns else None]
        encoding_map.append([col, val] + encoded_cols)

# Convert to DataFrame for display
encoding_df = pd.DataFrame(encoding_map, columns=["Original Column", "Original Value", "Encoded Column"])

# Plot categorical value distributions before encoding
fig, ax = plt.subplots(figsize=(10, 6))
df[categorical_columns].nunique().plot(kind='bar', ax=ax, color='skyblue')
ax.set_title("Number of Unique Values in Categorical Columns Before Encoding")
ax.set_ylabel("Unique Categories")
ax.set_xlabel("Categorical Columns")
plt.xticks(rotation=45)
plt.show()

# Display encoding table
encoding_df
