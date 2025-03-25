# analyze_imbalance.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../003_ModelComparison/raw.csv"
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
majority_classes = class_proportions[class_proportions >= threshold].index

# Group the minority classes into a single category "Other"
raw['class_grouped'] = raw['class'].apply(lambda x: x if x in majority_classes else 'Other')

# Plot the class distribution with Majority and Minority groups
plt.figure(figsize=(10, 6))

# Plot the majority and minority groups together
sns.countplot(x='class_grouped', data=raw, order=raw['class_grouped'].value_counts().index)
plt.title('Class Distribution: Majority vs Minority')
plt.xticks(rotation=45)
plt.show()

# Save this grouped dataset for further exploration
raw.to_csv('grouped_classes_dataset.csv', index=False)

# Print out the minority and majority classes for reference
print("\nMinority Classes (Below 5%):\n", minority_classes)
print("\nMajority Classes:\n", majority_classes)
