import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "../000_Data/raw.csv"
raw = pd.read_csv(file_path)

columns_to_convert = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

for column in columns_to_convert:
    raw[column] = pd.to_numeric(raw[column], errors='coerce')

# Check the data types
print(raw.dtypes)

# Select only numeric columns
numeric_columns = raw.select_dtypes(include=['float64', 'int64'])

#  if there are numeric columns available
if numeric_columns.empty:
    print("No numeric columns found in the dataset.")
else:
    # Calculate the corelation matrix
    correlation_matrix = numeric_columns.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()