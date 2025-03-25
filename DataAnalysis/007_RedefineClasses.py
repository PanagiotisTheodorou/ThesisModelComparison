import pandas as pd

# Load the dataset
file_path = "../ModelComparison/raw.csv"  # Replace with the actual path to your raw dataset
raw = pd.read_csv(file_path)

# 1. Count occurrences of each class
class_counts = raw['class'].value_counts()

# Display the counts of each class
print("Class Counts Before Transformation:")
print(class_counts)

# 2. Define the mapping of old class labels to new general class labels
class_mapping = {
    'A': 'Hyperthyroid', 'B': 'Hyperthyroid', 'C': 'Hyperthyroid', 'D': 'Hyperthyroid',
    'E': 'Hypothyroid', 'F': 'Hypothyroid', 'G': 'Hypothyroid', 'H': 'Hypothyroid',
    'I': 'Binding Protein Disorders', 'J': 'Binding Protein Disorders',
    'K': 'General Health',
    'L': 'Replacement Therapy', 'M': 'Replacement Therapy', 'N': 'Replacement Therapy',
    'O': 'Antithyroid Treatment', 'P': 'Antithyroid Treatment', 'Q': 'Antithyroid Treatment',
    'R': 'Miscellaneous', 'S': 'Miscellaneous', 'T': 'Miscellaneous',
    'OI': 'Antithyroid Treatment',
    'C|I': 'Hyperthyroid', 'H|K': 'Hypothyroid', 'AK': 'Hyperthyroid',
    'GK': 'Hypothyroid', 'GKJ': 'Hypothyroid', 'FK': 'Hypothyroid',
    'D|R': 'Hyperthyroid',
    'MK': 'None', 'KJ': 'None', 'LJ': 'None', 'MI': 'None', '-': 'None',
    'GI': 'Hypothyroid'
}

# 3. Apply the class transformation
raw['class_transformed'] = raw['class'].apply(lambda x: class_mapping.get(x, 'Unknown'))

# 4. Count occurrences of the transformed classes
transformed_class_counts = raw['class_transformed'].value_counts()

# Display the counts of the transformed classes
print("\nClass Counts After Transformation:")
print(transformed_class_counts)

# 5. Show the mapping to the user and ask for confirmation
print("\nClass Transformation Preview:")
for original, transformed in class_mapping.items():
    print(f"{original} -> {transformed}")

user_input = input("\nDo you want to proceed with saving the dataset with these new class labels? (yes/no): ").strip().lower()

if user_input == 'yes':
    # 6. Remove the original 'class' column and rename 'class_transformed' to 'class'
    raw = raw.drop(columns=['class'])
    raw = raw.rename(columns={'class_transformed': 'class'})

    # 7. Convert the 'class' column values to lowercase
    raw['class'] = raw['class'].str.lower()

    # Save the new dataset with the generalized classes
    output_file = "../data/raw_with_general_classes.csv"
    raw.to_csv(output_file, index=False)

    print(f"\nNew dataset saved as {output_file}")
else:
    print("\nOperation canceled. No changes were made.")
