import pandas as pd
import tkinter as tk
from tkinter import ttk

"""
    The code performs the following:
        1. loads the dataset
        2. Replaces '?' with NaN 
        3. Extracts column metadata (name, data type)
        4. Creates a tkinter ui
        5. Adds the records that depict the columns
"""

df = pd.read_csv("../000_Data/001_dataset_before_preprocessing.csv")

# Replace '?' with NaN to properly identify missing values
df.replace('?', pd.NA, inplace=True)

# Get column names, data types, missing value status, and count
column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Has Missing Values": ["Yes" if df[col].isnull().sum() > 0 else "No" for col in df.columns],
    "Missing Count": df.isnull().sum().astype(str)
})

row_count = len(column_info)
window_height = min(50 + row_count * 20, 800)

# main window
root = tk.Tk()
root.title("Column Data Types & Missing Values")
root.geometry(f"800x{window_height}")

# Create frame
frame = ttk.Frame(root, padding=10, borderwidth=2, relief="ridge")
frame.pack(expand=True, fill="both")

# Create table
tree = ttk.Treeview(frame, show="headings",
                     columns=["Column Name", "Data Type", "Has Missing Values", "Missing Count"])

# Define columns
tree.heading("Column Name", text="Column Name")
tree.heading("Data Type", text="Data Type")
tree.heading("Has Missing Values", text="Has Missing Values")
tree.heading("Missing Count", text="Missing Count")

tree.column("Column Name", width=220, anchor="center")
tree.column("Data Type", width=100, anchor="center")
tree.column("Has Missing Values", width=120, anchor="center")
tree.column("Missing Count", width=100, anchor="center")

style = ttk.Style()
style.configure("Treeview", font=("Arial", 9))
style.configure("Treeview.Heading", font=("Arial", 10, "bold"))

# Insert rows into the table
for _, row in column_info.iterrows():
    tree.insert("", "end", values=list(row))

tree.pack(expand=True, fill="both")

root.mainloop()