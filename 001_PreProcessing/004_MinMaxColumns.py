import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
    The code performs the following:
        1. Loads the dataset
        2. Selects te numerical columns
        3. calculates the min and max values 
        4. Shows the results in a box plot
"""

# Load the dataset
df = pd.read_csv('../000_Data/003_dataset_after_removing_outliers.csv')

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate min, mean, and max
summary_stats = numeric_columns.agg(['min', 'mean', 'max'])

# Print the results
print(summary_stats)

# Plot box-plots for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_columns)
plt.xticks(rotation=90)
plt.title("Box-plots for Numeric Columns (Outlier Detection)")
plt.show()
