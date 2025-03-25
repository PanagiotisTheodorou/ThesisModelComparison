import pandas as pd

"""
    The code performs the following: 
        1. Loads the dataset
        2. Calculate the 1st and 3rd quantiles for the age column (saw in the previous script that there are outliers)
        3. Calculate the difference between the 1st and 3rd quantile ranges (IQR)
        4. Uses the IQR to find outliers 
        5. Filters the outliers
        6. Saves a new Dataset without outliers
"""

# A quantile is a dividing point, that sets apart the distribution to sectors with similar probabilities

# Load dataset
df = pd.read_csv('../data/002_dataset_after_filling_missing_values.csv')

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
df_filtered.to_csv('../data/003_dataset_after_removing_outliers.csv', index=False)

print(f"Original dataset size: {df.shape[0]} rows")
print(f"Filtered dataset size: {df_filtered.shape[0]} rows")
print("Outliers in 'age' column removed successfully!")
