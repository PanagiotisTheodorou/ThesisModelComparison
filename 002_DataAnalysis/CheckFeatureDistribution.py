import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# Select numerical features
numerical_features = raw.select_dtypes(include=['float64', 'int64']).columns

# Plot histograms for each numerical feature
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(raw[feature], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
