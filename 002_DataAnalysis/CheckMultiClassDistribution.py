import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "../000_Data/raw.csv"
raw = pd.read_csv(file_path)

# show class distribution before transformetion
plt.figure(figsize=(10, 6))
sns.countplot(x='class', data=raw, order=raw['class'].value_counts().index)
plt.title('Distribution of Classes Before Transformation')
plt.xticks(rotation=45)
plt.show()