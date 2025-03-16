import pandas as pd

# Load dataset
df = pd.read_csv('ThyroxineData_Cleaned_v2.csv')

# Compute Q1 (25th percentile) and Q3 (75th percentile) for the Age column
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)

# Compute IQR (Interquartile Range)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_filtered = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

# Save the cleaned dataset
df_filtered.to_csv('ThyroxineData_Cleaned_NoOutliers.csv', index=False)

print(f"Original dataset size: {df.shape[0]} rows")
print(f"Filtered dataset size: {df_filtered.shape[0]} rows")
print("Outliers in 'age' column removed successfully!")
