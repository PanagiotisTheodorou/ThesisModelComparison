import customtkinter as ctk
from tkinter import scrolledtext, messagebox, Menu, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ThyroidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thyroid Diagnosis Predictor - Proof of Concept")
        self.root.geometry("1200x900")

        # Set CustomTkinter theme (light/dark)
        ctk.set_appearance_mode("System")  # "Light" or "Dark"
        ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

        # Initialize file_path as a StringVar
        self.file_path = ctk.StringVar()

        # Custom Header
        self.create_header()

        # Menu Bar
        self.create_menu()

        # Main Content Frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)

        # GUI Components
        self.create_widgets()

    def create_header(self):
        header = ctk.CTkFrame(self.root, height=100, corner_radius=0, fg_color="blue")
        header.pack(fill=ctk.X)

        # Title
        title = ctk.CTkLabel(header, text="Thyroid Diagnosis Predictor", font=("Arial", 24, "bold"), text_color="white")
        title.pack(side=ctk.LEFT, padx=20, pady=20)

        # Placeholder for Image
        self.header_image = ctk.CTkLabel(header, text="[Image Placeholder]", font=("Arial", 16, "italic"),
                                         text_color="white", width=150, height=80)
        self.header_image.pack(side=ctk.RIGHT, padx=20, pady=10)

        # Round edges on the bottom line
        header.grid_propagate(False)
        header.configure(corner_radius=20)

    def create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # File Menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.browse_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # View Menu
        view_menu = Menu(menubar, tearoff=0)
        view_menu.add_command(label="Show Results", command=self.show_results)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help Menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def show_about(self):
        messagebox.showinfo("About", "Thyroid Disease Model Comparison\nVersion 1.0\nDeveloped by Panayiotis Theodorou")

    def create_widgets(self):
        # File Path Input
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.pack(fill=ctk.X, pady=10)

        ctk.CTkLabel(input_frame, text="Dataset Path:").pack(side=ctk.LEFT, padx=10)
        self.file_entry = ctk.CTkEntry(input_frame, textvariable=self.file_path, width=400)
        self.file_entry.pack(side=ctk.LEFT, padx=10, expand=True, fill=ctk.X)
        ctk.CTkButton(input_frame, text="Browse", command=self.browse_file).pack(side=ctk.LEFT, padx=10)

        # Buttons
        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(fill=ctk.X, pady=10)

        ctk.CTkButton(button_frame, text="Load Data", command=self.load_data).pack(side=ctk.LEFT, padx=10)
        ctk.CTkButton(button_frame, text="Train Model", command=self.train_model).pack(side=ctk.LEFT, padx=10)
        ctk.CTkButton(button_frame, text="Show Results", command=self.show_results).pack(side=ctk.LEFT, padx=10)

        # Output Area
        self.output_area = scrolledtext.ScrolledText(self.main_frame, width=120, height=20, wrap=ctk.WORD)
        self.output_area.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Visualization Frame with Scrollbar
        self.visualization_frame = ctk.CTkFrame(self.main_frame)
        self.visualization_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Add a canvas and scrollbar to the visualization frame
        self.canvas = ctk.CTkCanvas(self.visualization_frame, bg="#f0f0f0")
        self.canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        scrollbar = ctk.CTkScrollbar(self.visualization_frame, orientation=ctk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=ctk.RIGHT, fill=ctk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create a frame inside the canvas to hold the visualizations
        self.visualization_container = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.visualization_container, anchor="nw")

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.file_path.set(file_path)

    def load_data(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a dataset file.")
            return

        try:
            # Simulate loading 000_Data
            self.output_area.insert(ctk.END, "Dataset loaded successfully!\n")
            self.output_area.insert(ctk.END, "Sample Data:\n")
            self.output_area.insert(ctk.END, "   age sex on_thyroxine  ...   TBG referral_source class\n")
            self.output_area.insert(ctk.END, "0   29   F            f  ...   NaN           other     -\n")
            self.output_area.insert(ctk.END, "1   29   F            f  ...   NaN           other     -\n")
            self.output_area.insert(ctk.END, "2   41   F            f  ...  11.0           other     -\n")
            self.output_area.insert(ctk.END, "3   36   F            f  ...  26.0           other     -\n")
            self.output_area.insert(ctk.END, "4   32   F            f  ...  36.0           other     S\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please load the dataset first.")
            return

        try:
            # Simulate training process
            self.output_area.insert(ctk.END, "Training model...\n")
            self.output_area.insert(ctk.END, "Filtering rare classes...\n")
            self.output_area.insert(ctk.END, "Applying dataset balancing...\n")
            self.output_area.insert(ctk.END, "Performing stratified train-test split...\n")
            self.output_area.insert(ctk.END, "Performing feature selection...\n")
            self.output_area.insert(ctk.END, "Selected Features: ['T3', 'FTI', 'TT4', 'T4U', 'TSH', 'TBG', 'age', 'on_thyroxine', 'referral_source', 'sex']\n")
            self.output_area.insert(ctk.END, "Hyperparameter tuning using GridSearchCV...\n")
            self.output_area.insert(ctk.END, "Best Parameters: {'bootstrap': False, 'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}\n")
            self.output_area.insert(ctk.END, "Model trained successfully!\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def show_results(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            # Simulate results
            self.output_area.insert(ctk.END, "Accuracy: 0.9451\n")
            self.output_area.insert(ctk.END, "Classification Report:\n")
            self.output_area.insert(ctk.END, "              precision    recall  f1-score   support\n")
            self.output_area.insert(ctk.END, "           -       0.99      0.98      0.98       255\n")
            self.output_area.insert(ctk.END, "           A       0.83      0.59      0.69        17\n")
            self.output_area.insert(ctk.END, "          AK       0.80      0.94      0.86        17\n")
            self.output_area.insert(ctk.END, "           B       0.90      1.00      0.95        18\n")
            self.output_area.insert(ctk.END, "           I       0.88      0.88      0.88        26\n")
            self.output_area.insert(ctk.END, "           J       0.94      0.94      0.94        18\n")
            self.output_area.insert(ctk.END, "           K       0.94      1.00      0.97        78\n")
            self.output_area.insert(ctk.END, "           L       0.82      0.78      0.80        18\n")
            self.output_area.insert(ctk.END, "           N       0.94      0.89      0.91        18\n")
            self.output_area.insert(ctk.END, "           R       0.82      0.85      0.84        27\n")
            self.output_area.insert(ctk.END, "           S       1.00      1.00      1.00        18\n\n")

            # Clear previous visualizations
            for widget in self.visualization_container.winfo_children():
                widget.destroy()

            # Section 1: Confusion Matrix
            self.output_area.insert(ctk.END, "Confusion Matrix:\n")
            self.plot_confusion_matrix()

            # Section 2: ROC Curve
            self.output_area.insert(ctk.END, "\nROC Curve:\n")
            self.plot_roc_curve()

            # Section 3: Feature Importance
            self.output_area.insert(ctk.END, "\nFeature Importance:\n")
            self.plot_feature_importance()

            # Section 4: Class Distribution
            self.output_area.insert(ctk.END, "\nClass Distribution:\n")
            self.plot_class_distribution()

            # Section 5: Correlation Heatmap
            self.output_area.insert(ctk.END, "\nCorrelation Heatmap:\n")
            self.plot_correlation_heatmap()

            # Section 6: Feature Distribution
            self.output_area.insert(ctk.END, "\nFeature Distribution:\n")
            self.plot_feature_distribution()

            # Section 7: Precision-Recall Curve
            self.output_area.insert(ctk.END, "\nPrecision-Recall Curve:\n")
            self.plot_precision_recall_curve()

            # Section 8: Class Label Mapping
            self.output_area.insert(ctk.END, "\nClass Label Mapping:\n")
            self.output_area.insert(ctk.END, "Letter\tDiagnosis\n")
            self.output_area.insert(ctk.END, "------\t---------\n")
            self.output_area.insert(ctk.END, "A\thyperthyroid\n")
            self.output_area.insert(ctk.END, "B\tT3 toxic\n")
            self.output_area.insert(ctk.END, "C\ttoxic goitre\n")
            self.output_area.insert(ctk.END, "D\tsecondary toxic\n")
            self.output_area.insert(ctk.END, "E\thypothyroid\n")
            self.output_area.insert(ctk.END, "F\tprimary hypothyroid\n")
            self.output_area.insert(ctk.END, "G\tcompensated hypothyroid\n")
            self.output_area.insert(ctk.END, "H\tsecondary hypothyroid\n")
            self.output_area.insert(ctk.END, "I\tincreased binding protein\n")
            self.output_area.insert(ctk.END, "J\tdecreased binding protein\n")
            self.output_area.insert(ctk.END, "K\tconcurrent non-thyroidal illness\n")
            self.output_area.insert(ctk.END, "L\tconsistent with replacement therapy\n")
            self.output_area.insert(ctk.END, "M\tunderreplaced\n")
            self.output_area.insert(ctk.END, "N\toverreplaced\n")
            self.output_area.insert(ctk.END, "O\tantithyroid drugs\n")
            self.output_area.insert(ctk.END, "P\tI131 treatment\n")
            self.output_area.insert(ctk.END, "Q\tsurgery\n")
            self.output_area.insert(ctk.END, "R\tdiscordant assay results\n")
            self.output_area.insert(ctk.END, "S\televated TBG\n")
            self.output_area.insert(ctk.END, "T\televated thyroid hormones\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show results: {e}")



if __name__ == "__main__":
    root = ctk.CTk()
    app = ThyroidApp(root)
    root.mainloop()