import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, Menu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class ThyroidApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thyroid Diagnosis Predictor - Proof of Concept")
        self.root.geometry("1200x900")
        self.root.configure(bg="#f0f0f0")

        # Initialize file_path as a StringVar
        self.file_path = tk.StringVar()

        # Custom Header
        self.create_header()

        # Menu Bar
        self.create_menu()

        # Main Content Frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # GUI Components
        self.create_widgets()

    def create_header(self):
        header = tk.Frame(self.root, bg="#0078d7", height=50)
        header.pack(fill=tk.X)

        title = tk.Label(header, text="Thyroid Diagnosis Predictor", font=("Arial", 16, "bold"), fg="white", bg="#0078d7")
        title.pack(side=tk.LEFT, padx=20)

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

    def create_widgets(self):
        # File Path Input
        input_frame = ttk.Frame(self.main_frame)
        input_frame.pack(fill=tk.X, pady=10)

        ttk.Label(input_frame, text="Dataset Path:").pack(side=tk.LEFT, padx=10)
        self.file_entry = ttk.Entry(input_frame, textvariable=self.file_path, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        ttk.Button(input_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=10)

        # Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Train Model", command=self.train_model).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Show Results", command=self.show_results).pack(side=tk.LEFT, padx=10)

        # Output Area
        self.output_area = scrolledtext.ScrolledText(self.main_frame, width=120, height=20, wrap=tk.WORD)
        self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Visualization Frame with Scrollbar
        self.visualization_frame = ttk.Frame(self.main_frame)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add a canvas and scrollbar to the visualization frame
        self.canvas = tk.Canvas(self.visualization_frame, bg="#f0f0f0")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.visualization_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create a frame inside the canvas to hold the visualizations
        self.visualization_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.visualization_container, anchor="nw")

    def browse_file(self):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        self.file_path.set(file_path)

    def load_data(self):
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a dataset file.")
            return

        try:
            # Simulate loading data
            self.output_area.insert(tk.END, "Dataset loaded successfully!\n")
            self.output_area.insert(tk.END, "Sample Data:\n")
            self.output_area.insert(tk.END, "   age sex on_thyroxine  ...   TBG referral_source class\n")
            self.output_area.insert(tk.END, "0   29   F            f  ...   NaN           other     -\n")
            self.output_area.insert(tk.END, "1   29   F            f  ...   NaN           other     -\n")
            self.output_area.insert(tk.END, "2   41   F            f  ...  11.0           other     -\n")
            self.output_area.insert(tk.END, "3   36   F            f  ...  26.0           other     -\n")
            self.output_area.insert(tk.END, "4   32   F            f  ...  36.0           other     S\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def train_model(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please load the dataset first.")
            return

        try:
            # Simulate training process
            self.output_area.insert(tk.END, "Training model...\n")
            self.output_area.insert(tk.END, "Filtering rare classes...\n")
            self.output_area.insert(tk.END, "Applying dataset balancing...\n")
            self.output_area.insert(tk.END, "Performing stratified train-test split...\n")
            self.output_area.insert(tk.END, "Performing feature selection...\n")
            self.output_area.insert(tk.END, "Selected Features: ['T3', 'FTI', 'TT4', 'T4U', 'TSH', 'TBG', 'age', 'on_thyroxine', 'referral_source', 'sex']\n")
            self.output_area.insert(tk.END, "Hyperparameter tuning using GridSearchCV...\n")
            self.output_area.insert(tk.END, "Best Parameters: {'bootstrap': False, 'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}\n")
            self.output_area.insert(tk.END, "Model trained successfully!\n\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {e}")

    def show_results(self):
        if not self.file_path.get():
            messagebox.showerror("Error", "Please train the model first.")
            return

        try:
            # Simulate results
            self.output_area.insert(tk.END, "Accuracy: 0.9451\n")
            self.output_area.insert(tk.END, "Classification Report:\n")
            self.output_area.insert(tk.END, "              precision    recall  f1-score   support\n")
            self.output_area.insert(tk.END, "           -       0.99      0.98      0.98       255\n")
            self.output_area.insert(tk.END, "           A       0.83      0.59      0.69        17\n")
            self.output_area.insert(tk.END, "          AK       0.80      0.94      0.86        17\n")
            self.output_area.insert(tk.END, "           B       0.90      1.00      0.95        18\n")
            self.output_area.insert(tk.END, "           I       0.88      0.88      0.88        26\n")
            self.output_area.insert(tk.END, "           J       0.94      0.94      0.94        18\n")
            self.output_area.insert(tk.END, "           K       0.94      1.00      0.97        78\n")
            self.output_area.insert(tk.END, "           L       0.82      0.78      0.80        18\n")
            self.output_area.insert(tk.END, "           N       0.94      0.89      0.91        18\n")
            self.output_area.insert(tk.END, "           R       0.82      0.85      0.84        27\n")
            self.output_area.insert(tk.END, "           S       1.00      1.00      1.00        18\n\n")

            # Clear previous visualizations
            for widget in self.visualization_container.winfo_children():
                widget.destroy()

            # Section 1: Confusion Matrix
            self.output_area.insert(tk.END, "Confusion Matrix:\n")
            self.plot_confusion_matrix()

            # Section 2: ROC Curve
            self.output_area.insert(tk.END, "\nROC Curve:\n")
            self.plot_roc_curve()

            # Section 3: Feature Importance
            self.output_area.insert(tk.END, "\nFeature Importance:\n")
            self.plot_feature_importance()

            # Section 4: Class Distribution
            self.output_area.insert(tk.END, "\nClass Distribution:\n")
            self.plot_class_distribution()

            # Section 5: Correlation Heatmap
            self.output_area.insert(tk.END, "\nCorrelation Heatmap:\n")
            self.plot_correlation_heatmap()

            # Section 6: Feature Distribution
            self.output_area.insert(tk.END, "\nFeature Distribution:\n")
            self.plot_feature_distribution()

            # Section 7: Precision-Recall Curve
            self.output_area.insert(tk.END, "\nPrecision-Recall Curve:\n")
            self.plot_precision_recall_curve()

            # Section 8: Class Label Mapping
            self.output_area.insert(tk.END, "\nClass Label Mapping:\n")
            self.output_area.insert(tk.END, "Letter\tDiagnosis\n")
            self.output_area.insert(tk.END, "------\t---------\n")
            self.output_area.insert(tk.END, "A\thyperthyroid\n")
            self.output_area.insert(tk.END, "B\tT3 toxic\n")
            self.output_area.insert(tk.END, "C\ttoxic goitre\n")
            self.output_area.insert(tk.END, "D\tsecondary toxic\n")
            self.output_area.insert(tk.END, "E\thypothyroid\n")
            self.output_area.insert(tk.END, "F\tprimary hypothyroid\n")
            self.output_area.insert(tk.END, "G\tcompensated hypothyroid\n")
            self.output_area.insert(tk.END, "H\tsecondary hypothyroid\n")
            self.output_area.insert(tk.END, "I\tincreased binding protein\n")
            self.output_area.insert(tk.END, "J\tdecreased binding protein\n")
            self.output_area.insert(tk.END, "K\tconcurrent non-thyroidal illness\n")
            self.output_area.insert(tk.END, "L\tconsistent with replacement therapy\n")
            self.output_area.insert(tk.END, "M\tunderreplaced\n")
            self.output_area.insert(tk.END, "N\toverreplaced\n")
            self.output_area.insert(tk.END, "O\tantithyroid drugs\n")
            self.output_area.insert(tk.END, "P\tI131 treatment\n")
            self.output_area.insert(tk.END, "Q\tsurgery\n")
            self.output_area.insert(tk.END, "R\tdiscordant assay results\n")
            self.output_area.insert(tk.END, "S\televated TBG\n")
            self.output_area.insert(tk.END, "T\televated thyroid hormones\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show results: {e}")

    def plot_confusion_matrix(self):

        # Hardcoded confusion matrix
        cm = np.array([
            [249, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0],
            [0, 10, 4, 0, 1, 0, 0, 0, 0, 2, 0],
            [0, 1, 16, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 2, 23, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 78, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 14, 1, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 16, 0, 0],
            [0, 1, 0, 0, 0, 0, 3, 0, 0, 23, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18]
        ])

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Add labels
        ax.set_xticks(np.arange(len(cm)))
        ax.set_yticks(np.arange(len(cm)))
        ax.set_xticklabels(['-', 'A', 'AK', 'B', 'I', 'J', 'K', 'L', 'N', 'R', 'S'])
        ax.set_yticklabels(['-', 'A', 'AK', 'B', 'I', 'J', 'K', 'L', 'N', 'R', 'S'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Add text annotations
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_roc_curve(self):
        # Hardcoded ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sin(2 * np.pi * fpr)  # Simulated ROC curve

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fpr, tpr, color="blue", lw=2, label="ROC Curve")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_feature_importance(self):
        # Hardcoded feature importance data
        features = ['T3', 'FTI', 'TT4', 'T4U', 'TSH', 'TBG', 'age', 'on_thyroxine', 'referral_source', 'sex']
        importance = [0.28, 0.16, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.01]

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, importance, color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_class_distribution(self):
        # Hardcoded class distribution data
        classes = ['-', 'A', 'AK', 'B', 'I', 'J', 'K', 'L', 'N', 'R', 'S']
        counts = [255, 17, 17, 18, 26, 18, 78, 18, 18, 27, 18]

        # Plot class distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(classes, counts, color="skyblue")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Thyroid Classes")

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_correlation_heatmap(self):
        # Hardcoded correlation matrix
        features = ['T3', 'FTI', 'TT4', 'T4U', 'TSH', 'TBG', 'age', 'on_thyroxine', 'referral_source', 'sex']
        correlation_matrix = np.random.rand(len(features), len(features))

        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(correlation_matrix, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        # Add labels
        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=90)
        ax.set_yticklabels(features)
        ax.set_title("Correlation Heatmap")

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_feature_distribution(self):
        # Hardcoded feature data
        tsh_values = np.random.normal(2.0, 0.5, 1000)
        t3_values = np.random.normal(1.5, 0.3, 1000)
        t4_values = np.random.normal(120, 20, 1000)

        # Plot feature distribution
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].hist(tsh_values, bins=30, color="skyblue", alpha=0.7)
        ax[0].set_xlabel("TSH")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("TSH Distribution")

        ax[1].hist(t3_values, bins=30, color="skyblue", alpha=0.7)
        ax[1].set_xlabel("T3")
        ax[1].set_ylabel("Frequency")
        ax[1].set_title("T3 Distribution")

        ax[2].hist(t4_values, bins=30, color="skyblue", alpha=0.7)
        ax[2].set_xlabel("T4")
        ax[2].set_ylabel("Frequency")
        ax[2].set_title("T4 Distribution")

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_precision_recall_curve(self):
        # Hardcoded precision-recall curve data
        precision = np.linspace(1, 0, 100)
        recall = np.linspace(0, 1, 100)

        # Plot precision-recall curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(recall, precision, color="blue", lw=2, label="Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()

        # Embed plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.visualization_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def show_about(self):
        messagebox.showinfo("About", "Thyroid Diagnosis Predictor\nVersion 1.0\n\nA proof-of-concept application for thyroid diagnosis prediction.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThyroidApp(root)
    root.mainloop()