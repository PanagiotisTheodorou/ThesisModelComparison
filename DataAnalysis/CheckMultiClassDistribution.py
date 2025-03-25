# load_and_visualize.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# Plot the class distribution before transformation
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=raw, order=raw['class'].value_counts().index)
plt.title('Distribution of Classes Before Transformation')
plt.xticks(rotation=45)
plt.show()
