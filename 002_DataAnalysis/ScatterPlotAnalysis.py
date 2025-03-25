import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../003_ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# Convert relevant columns to numeric (adjust the column names as needed)
columns_to_convert = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']  # Add other relevant columns here

for column in columns_to_convert:
    raw[column] = pd.to_numeric(raw[column], errors='coerce')

# Select numerical features again after conversion
numerical_features = raw.select_dtypes(include=['float64', 'int64']).columns

# Check the numerical features now
print(f'Numerical features: {numerical_features}')

# Create scatter plots for pairwise relationships between numerical features
for i in range(len(numerical_features)):
    for j in range(i + 1, len(numerical_features)):
        try:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=raw[numerical_features[i]], y=raw[numerical_features[j]], hue=raw['class'],
                            palette='Set2')
            plt.title(f'Scatter Plot of {numerical_features[i]} vs {numerical_features[j]}')
            plt.xlabel(numerical_features[i])
            plt.ylabel(numerical_features[j])

            # Adjust legend to the right of the plot
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.show()
        except Exception as e:
            print(f"Error plotting {numerical_features[i]} vs {numerical_features[j]}: {e}")
