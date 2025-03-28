import pandas as pd

"""
    The code performs the following:
        1. Loads the dataset
        2. defines the mapping between old and new classes
        3. provides the usr with the option to take the new classes
        4. Based on user input, it generates the new classes
"""

file_path = "../003_ModelComparison/raw.csv"
raw = pd.read_csv(file_path)

# count ocurrences of each class
class_counts = raw['class'].value_counts()

print("Class Counts Before Transformation:")
print(class_counts)

# mapping - old vs new classes
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

# apply the class transformation
raw['class_transformed'] = raw['class'].apply(lambda x: class_mapping.get(x, 'Unknown'))

# Cout occurrences of the transformed classes
transformed_class_counts = raw['class_transformed'].value_counts()

print("\nClass Counts After Transformation:")
print(transformed_class_counts)

print("\nClass Transformation Preview:")
for original, transformed in class_mapping.items():
    print(f"{original} -> {transformed}")

user_input = input("\nDo you want to proceed with saving the dataset with these new class labels? (yes/no): ").strip().lower()

if user_input == 'yes':
    raw = raw.drop(columns=['class'])
    raw = raw.rename(columns={'class_transformed': 'class'})

    raw['class'] = raw['class'].str.lower()

    output_file = "../000_Data/raw_with_general_classes.csv"
    raw.to_csv(output_file, index=False)

    print(f"\nNew dataset saved as {output_file}")
else:
    print("\nOperation canceled. No changes were made.")
