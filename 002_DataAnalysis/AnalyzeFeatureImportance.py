import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

file_path = "../000_Data/raw.csv"
raw = pd.read_csv(file_path)

# get categrical columns
categorical_columns = raw.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()

# apply label encoding for categorical columns
for col in categorical_columns:
    raw[col] = label_encoder.fit_transform(raw[col])

# Prepare the data
# X = features
# y = target class

X = raw.drop(columns=['class'])
y = raw['class']

# Split the data into training and testig
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importances = model.feature_importances_

#make a Data Frame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()
