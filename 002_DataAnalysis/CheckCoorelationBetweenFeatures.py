import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# Convert all columns that should be numeric to numeric, coercing errors to NaN
columns_to_convert = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']  # Add other relevant columns here

for column in columns_to_convert:
    raw[column] = pd.to_numeric(raw[column], errors='coerce')

# Check the data types again
print(raw.dtypes)

# Select only numeric columns
numeric_columns = raw.select_dtypes(include=['float64', 'int64'])

# Check if there are numeric columns available
if numeric_columns.empty:
    print("No numeric columns found in the dataset.")
else:
    # Calculate the correlation matrix for numeric columns
    correlation_matrix = numeric_columns.corr()

    # Plot the correlation matrix using a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
