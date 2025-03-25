# analyze_minority_only.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# Class counts and proportions
class_counts = raw['class'].value_counts()
class_proportions = raw['class'].value_counts(normalize=True)

# Print class counts and proportions
print("Class Counts:\n", class_counts)
print("\nClass Proportions:\n", class_proportions)

# Identify minority classes (classes < 5% of the dataset)
threshold = 0.05
minority_classes = class_proportions[class_proportions < threshold].index

# Filter the dataset to include only minority classes
minority_data = raw[raw['class'].isin(minority_classes)]

# Plot the class distribution of minority classes
plt.figure(figsize=(10, 6))

# Plot the minority classes only
sns.countplot(x='class', data=minority_data, order=minority_data['class'].value_counts().index)
plt.title('Minority Class Distribution (Excluding Majority Classes)')
plt.xticks(rotation=45)
plt.show()

# Save this filtered dataset for further exploration
minority_data.to_csv('minority_classes_only_dataset.csv', index=False)

# Print out the minority classes for reference
print("\nMinority Classes (Below 5%):\n", minority_classes)
