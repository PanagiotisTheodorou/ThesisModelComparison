import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ThyroxineData_Cleaned_NoOutliers.csv')

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number'])

# Calculate min, mean, and max
summary_stats = numeric_columns.agg(['min', 'mean', 'max'])

# Print the results
print(summary_stats)

# Plot boxplots for outlier detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_columns)
plt.xticks(rotation=90)
plt.title("Boxplots for Numeric Columns (Outlier Detection)")
plt.show()
