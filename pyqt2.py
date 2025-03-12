import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QTabWidget,
    QHBoxLayout, QFileDialog, QMenuBar, QMenu, QTableWidget, QTableWidgetItem, QSpinBox,
    QTextEdit, QProgressDialog, QDialog, QGridLayout, QFrame, QSizePolicy
)
from PyQt6.QtGui import QFont, QAction, QMovie
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, f1_score, \
    roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from colorama import colorama_text, Fore, Style


# Global dataframe
df = pd.DataFrame()


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class LoadingWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Processing...")
        self.setFixedSize(250, 250)  # Set window size
        self.setWindowModality(Qt.WindowModality.ApplicationModal)  # Blocks interactions with the main UI

        layout = QVBoxLayout()

        # QLabel for the GIF animation
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load the GIF animation
        self.loading_movie = QMovie("load_indicator.gif")
        self.loading_label.setMovie(self.loading_movie)
        self.loading_movie.start()  # Start animation

        # Message to the user
        self.message_label = QLabel("Building the Machine Learning model...\nPlease wait.", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add widgets to layout
        layout.addWidget(self.loading_label)
        layout.addWidget(self.message_label)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Trainer")
        self.setGeometry(100, 100, 800, 600)

        # Copy of global df
        self.df: pd.DataFrame = df.copy()

        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Header Section
        header_frame = QWidget()
        header_layout = QVBoxLayout()
        header_frame.setStyleSheet("background-color: lightblue; padding: 10px;")
        header_label = QLabel("Machine Learning Model Trainer", self)
        header_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header_label)
        header_frame.setLayout(header_layout)
        main_layout.addWidget(header_frame)

        # Menu Bar Below Header
        menu_bar = QMenuBar()
        theme_menu = QMenu("Theme", self)
        menu_bar.addMenu(theme_menu)

        light_theme_action = QAction("Light Theme", self)
        dark_theme_action = QAction("Dark Theme", self)

        light_theme_action.triggered.connect(self.set_light_theme)
        dark_theme_action.triggered.connect(self.set_dark_theme)

        theme_menu.addAction(light_theme_action)
        theme_menu.addAction(dark_theme_action)
        main_layout.addWidget(menu_bar)

        # Operations Section
        operations_layout = QHBoxLayout()

        self.import_button = QPushButton("Import Data")
        self.import_button.clicked.connect(self.import_file)
        self.import_button.setFixedSize(150, 50)
        self.import_button.setFont(QFont("Arial", 12))
        self.import_button.setStyleSheet("border: 2px solid #0078D7; border-radius: 10px; background-color: #E1F5FE;")
        operations_layout.addWidget(self.import_button)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.perform_processing)
        self.process_button.setFixedSize(150, 50)
        self.process_button.setFont(QFont("Arial", 12))
        self.process_button.setStyleSheet("border: 2px solid #FF9800; border-radius: 10px; background-color: #FFF3E0;")
        operations_layout.addWidget(self.process_button)

        main_layout.addLayout(operations_layout)

        # File Path Display
        self.file_path_label = QLabel("No file loaded.")
        self.file_path_label.setStyleSheet("font-style: italic;")
        main_layout.addWidget(self.file_path_label)

        # Tab Widget for Data, Logs, Results, and Charts
        self.tabs = QTabWidget()

        # Data Tab
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)
        self.data_table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)  # Ensure table resizes
        self.data_table.setStyleSheet("""
                QHeaderView::section {
                    background-color: lightblue;
                    padding: 4px;
                    border: 1px solid #6c6c6c;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        data_layout.addWidget(self.data_table)
        self.row_spinbox = QSpinBox()
        self.row_spinbox.setMinimum(1)
        self.row_spinbox.setMaximum(100)
        self.row_spinbox.setValue(10)
        data_layout.addWidget(QLabel("Number of rows to display:"))
        data_layout.addWidget(self.row_spinbox)
        self.data_tab.setLayout(data_layout)

        # Logs Tab
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout()
        self.logs_tab.setLayout(logs_layout)

        # Add a QTextEdit for logging messages
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)  # Make it read-only
        logs_layout.addWidget(self.log_text_edit)

        # Results Tab
        self.results_tab = QWidget()
        results_layout = QGridLayout()
        self.results_tab.setLayout(results_layout)

        # Create 6 sections (2 columns x 3 rows)
        self.chart_widgets = []  # Store matplotlib canvases

        for row in range(3):
            for col in range(2):
                chart_frame = QFrame()
                chart_frame.setFrameShape(QFrame.Shape.Box)
                chart_frame.setFixedSize(350, 250)  # Adjust size as needed
                layout = QVBoxLayout(chart_frame)

                # Placeholder title
                chart_title = QLabel(f"Chart {row * 2 + col + 1}")
                chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(chart_title)

                # Create Matplotlib figure only for the first section
                if row == 0 and col == 0:
                    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                    from matplotlib.figure import Figure

                    fig = Figure(figsize=(3, 2))
                    ax = fig.add_subplot(111)
                    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], marker='o')  # Dummy data
                    ax.set_title("Sample Chart")

                    canvas = FigureCanvas(fig)
                    layout.addWidget(canvas)
                    self.chart_widgets.append(canvas)

                results_layout.addWidget(chart_frame, row, col)

        # allocate tabs
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.results_tab, "Results")

        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.set_light_theme()

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def print_message(self, message, color):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = getattr(Fore, color.upper(), Fore.WHITE)  # Default to WHITE if color is invalid
        print(color_code + f"[{timestamp}]\n{message}\n" + Style.RESET_ALL)

    def print_message_extended(self, message, color):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = getattr(Fore, color.upper(), Fore.WHITE)  # Default to WHITE if color is invalid
        print(color_code + f"[{timestamp}]\n{message}" + Style.RESET_ALL)

    # Functions that regard the functionality of the model

    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            global df
            df = self.load_data(file_path)
            self.df = df.copy()
            self.file_path_label.setText(f"Loaded: {file_path}")
            self.display_data()

    def load_data(self, file_path):
        self.log_message("Loading dataset...")
        self.print_message("Loading dataset...", "GREEN")
        df = pd.read_csv(file_path, na_values='?')
        return df

    def display_data(self):
        if self.df.empty:
            return

        progress = QProgressDialog("Loading data...", None, 0, len(self.df), self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(500)  # Show after 500ms

        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setRowCount(len(self.df))
        self.data_table.setHorizontalHeaderLabels(self.df.columns)

        for i, row in self.df.iterrows():
            progress.setValue(i)
            QApplication.processEvents()  # Keep the UI responsive
            for j, cell in enumerate(row):
                self.data_table.setItem(i, j, QTableWidgetItem(str(cell)))

        progress.setValue(len(self.df))  # Ensure progress completes

        self.print_message("Dataset imported successfully", "LIGHTGREEN_EX")
        self.print_message("Dataset loaded successfully", "LIGHTGREEN_EX")
        self.log_message("Dataset imported successfully")
        self.print_message_extended("Dataset Head", "YELLOW")
        print(df.head())

    def add_chart_to_results_tab(self, fig, row, col):
        """
        Adds a matplotlib figure to the Results tab at the specified grid position.
        Args:
            fig (matplotlib.figure.Figure): The figure to embed.
            row (int): The row index in the grid layout.
            col (int): The column index in the grid layout.
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        # Create a canvas to embed the figure
        canvas = FigureCanvas(fig)

        # Remove any existing widget in the specified grid position
        existing_widget = self.results_tab.layout().itemAtPosition(row, col).widget()
        if existing_widget:
            existing_widget.deleteLater()

        # Add the canvas to the specified grid position
        self.results_tab.layout().addWidget(canvas, row, col)
        canvas.draw()  # Render the plot


    def perform_processing(self):

        self.log_message("Initializing data processing...")

        # Show the loading window
        self.loading_window = LoadingWindow()
        self.loading_window.show()

        # Ensure the UI updates before running the processing
        QApplication.processEvents()

        try:
            target_column = 'class'

            print(Fore.YELLOW + "\nPerforming data processing" + Style.RESET_ALL)
            self.log_message("Performing data processing")

            # Step 1: Remove unwanted columns

            self.log_message("Removing unwanted columns")
            self.df, columns_to_drop = remove_unwanted_columns(df)
            self.log_message(f"Dropped columns: {columns_to_drop}")

            # Step 2: Remove outliers
            self.log_message("Removing outliers")
            self.df, df_no_outliers = remove_outliers(df)
            self.log_message(f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped.")

            # Step 3: Fill missing values
            self.log_message("Filling missing values")
            self.df = fill_missing_values(df)
            self.log_message("Missing values filled.")

            # Step 4: Encode categorical variables
            self.log_message("Encoding categorical variables")
            self.df, self.label_encoders, self.label_mappings = encode_categorical(df)
            self.log_message("Categorical columns encoded.")

            self.log_message("Starting model training")
            self.log_message("Filtering rare classes")
            self.log_message("Applying dataset balancing")
            self.log_message("Dataset balanced.")
            self.log_message("Performing stratified train-test split")
            self.log_message("Hyperparameter tuning using GridSearchCV")
            model, X_train, X_test, y_train, y_test, acr, best_params, cr = train_model(df, target_column, self.label_mappings)
            self.log_message(f"Accuracy: {acr:.4f}")
            self.log_message(f"Best Parameters: {str(best_params)}")

            # Check for overfitting
            self.log_message("Checking for Overfitting")
            train_acc, test_acc, cross_acc, cross_std = check_overfitting(model, X_train, y_train, X_test, y_test)
            self.log_message(f"Training Accuracy: {train_acc:.4f}")
            self.log_message(f"Testing Accuracy: {test_acc:.4f}")
            self.log_message(f"Cross-Validation Accuracy: {cross_acc:.4f} +/- {cross_std:.4f}")
            # Check for overfitting
            if train_acc > test_acc + 0.05:
                self.log_message("Possible Overfitting Detected!")
            else:
                self.log_message("No significant overfitting detected.")

            # Construct and display confusion matrix and additional metrics in the console
            construct_confussion_matrix(model, X_test, y_test, self.label_mappings, model_name="Random Forest")

            # Plot ROC curve and embed it in the Results tab
            self.log_message("Generating ROC Curve")
            roc_fig = plot_roc_auc(model, X_test, y_test)
            self.add_chart_to_results_tab(roc_fig, row=0, col=0)  # Add to the first chart section

            self.display_data()
            print(Fore.LIGHTYELLOW_EX + "Data processing complete!\n" + Style.RESET_ALL)
            self.log_message("Data processing complete!")
        except Exception as e:
            self.log_message(f"Error: {str(e)}")

        finally:
            # Close the loading window when processing is done
            self.loading_window.close()

    def set_light_theme(self):
        self.setStyleSheet("background-color: white; color: black;")

    def set_dark_theme(self):
        self.setStyleSheet("background-color: #2E2E2E; color: white;")


def remove_unwanted_columns(df):
    print(Fore.GREEN + "\nRemoving unwanted columns" + Style.RESET_ALL)
    columns_to_drop = [col for col in df.columns if 'measured' in col.lower()]
    df.drop(columns=columns_to_drop, inplace=True)
    print(Fore.LIGHTGREEN_EX + f"Dropped columns: {columns_to_drop}\n" + Style.RESET_ALL)
    return df, columns_to_drop


def remove_outliers(df):
    print(Fore.GREEN + "\nRemoving outliers..." + Style.RESET_ALL)
    numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    Q1 = df[numerical_columns].quantile(0.25)
    Q3 = df[numerical_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]
    print(Fore.LIGHTGREEN_EX + f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped." + Style.RESET_ALL)
    return df, df_no_outliers


def fill_missing_values(df):
    print(Fore.GREEN + "\nFilling missing values..." + Style.RESET_ALL)
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].mean())
    print(Fore.LIGHTGREEN_EX + "Missing values filled.\n" + Style.RESET_ALL)
    return df


def encode_categorical(df):
    print(Fore.GREEN + "\nEncoding categorical variables..." + Style.RESET_ALL)
    label_encoders = {}
    label_mappings = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(enumerate(le.classes_))
    print(Fore.LIGHTGREEN_EX + "Categorical columns encoded.\n" + Style.RESET_ALL)
    return df, label_encoders, label_mappings

def balance_dataset(df, target_column):
    """
        Function to balance dataset
        TODO add explanation for the formula
    """
    print(Fore.GREEN + "\nBalancing dataset by oversampling minority classes proportionally..." + Style.RESET_ALL)

    # Count occurrences of each class
    class_counts = df[target_column].value_counts()
    majority_class = class_counts.idxmax()
    minority_classes = class_counts[class_counts.index != majority_class].index

    # Compute total occurrences of all minority classes
    total_minority_occurrences = class_counts[minority_classes].sum()
    num_minority_classes = len(minority_classes)

    # Determine target occurrences for each minority class
    target_counts = total_minority_occurrences // num_minority_classes
    sampling_strategy = {}

    for cls in minority_classes:
        if class_counts[cls] < target_counts:
            sampling_strategy[cls] = target_counts

    # Apply SMOTE for oversampling
    X = df.drop(columns=[target_column])
    y = df[target_column]
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled

    # Apply equal sampling of majority class instances
    final_data = []
    for cls in minority_classes:
        minority_samples = resampled_df[resampled_df[target_column] == cls]
        majority_samples = resampled_df[resampled_df[target_column] == majority_class].sample(n=len(minority_samples),
                                                                                              random_state=42)
        final_data.append(minority_samples)
        final_data.append(majority_samples)

    balanced_df = pd.concat(final_data).sample(frac=1, random_state=42).reset_index(drop=True)

    print(Fore.LIGHTGREEN_EX + "Dataset balanced.\n" + Style.RESET_ALL)

    return balanced_df


def feature_selection(X_train, y_train, X_test, threshold=0.01):
    """
    Perform feature selection using Random Forest feature importance.
    Features with importance greater than the threshold are selected.
    """
    print(Fore.GREEN + "\nPerforming feature selection..." + Style.RESET_ALL)

    # Train a Random Forest model to get feature importance
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display feature importances
    print(Fore.CYAN + "\nFeature Importances:" + Style.RESET_ALL)
    print(feature_importance_df)

    # Select features with importance greater than the threshold
    selected_features = feature_importance_df[feature_importance_df["Importance"] > threshold]["Feature"].tolist()

    print(Fore.LIGHTGREEN_EX + f"\nSelected Features (Importance > {threshold}):" + Style.RESET_ALL)
    print(selected_features)

    # Filter the datasets to include only selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    return X_train_selected, X_test_selected, feature_importance_df


def train_model(df, target_column, label_mappings):
    """
            Function to train the model, Steps:
            1. Filter rare classes, and for those apply the balancing logic
            2. split between dependent and non dependent column
            3. Split the dataset to test and train
            4. Create a parameter grid for hyperparameter tuning
            5. Create and Train the model
            6. Apply Grid search so that the interpreter will loop throygh the available parameters and find the best ones
            7. Print out the best model, and the statistics, then return the chosen model
        """

    print(Fore.GREEN + "\nTraining model..." + Style.RESET_ALL)

    print(Fore.LIGHTGREEN_EX + "Filtering rare classes..." + Style.RESET_ALL)

    class_counts = df[target_column].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df[target_column].isin(valid_classes)]

    print(Fore.LIGHTGREEN_EX + "Applying dataset balancing..." + Style.RESET_ALL)

    df = balance_dataset(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(Fore.LIGHTGREEN_EX + "Performing stratified train-test split..." + Style.RESET_ALL)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Perform feature selection
    # TODO: I can uncomment and comment this in order to add feature selection
    #X_train, X_test, feature_importance_df = feature_selection(X_train, y_train, X_test, threshold=0.01)

    print(Fore.LIGHTGREEN_EX + "Hyperparameter tuning using GridSearchCV..." + Style.RESET_ALL)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(Fore.LIGHTGREEN_EX + f"Best Parameters: {grid_search.best_params_}" + Style.RESET_ALL)

    best_model = grid_search.best_estimator_

    predictions = best_model.predict(X_test)

    accuracy_scr = accuracy_score(y_test, predictions)

    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy_score(y_test, predictions):.4f}" + Style.RESET_ALL)

    # Decode numeric labels back to original class labels
    y_test_decoded = decode_predictions(pd.Series(y_test), label_mappings, target_column)
    predictions_decoded = decode_predictions(pd.Series(predictions), label_mappings, target_column)

    # Print classification report with decoded labels
    print(Fore.LIGHTGREEN_EX + "Classification Report:\n" + Style.RESET_ALL)
    cr_dict = classification_report(y_test_decoded, predictions_decoded, output_dict=True)
    print(classification_report(y_test_decoded, predictions_decoded))

    return best_model, X_train, X_test, y_train, y_test, accuracy_scr, grid_search.best_params_, cr_dict


def decode_predictions(predictions, label_mappings, column_name):
    """
    Convert numerical predictions back to categorical labels.
    """
    return predictions.map(label_mappings[column_name])


def check_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Function to check for overfitting by comparing training and test accuracy.
    Also performs cross-validation to verify model generalization.
    """
    print(Fore.GREEN + "\nChecking for Overfitting..." + Style.RESET_ALL)

    # Predict on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(Fore.LIGHTGREEN_EX + f"Training Accuracy: {train_acc:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Test Accuracy: {test_acc:.4f}" + Style.RESET_ALL)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    cross_acc = cv_scores.mean()
    cross_std = cv_scores.std()
    print(Fore.LIGHTGREEN_EX + f"Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}" + Style.RESET_ALL)

    # Check for overfitting
    if train_acc > test_acc + 0.05:  # If train accuracy is much higher than test accuracy
        print(Fore.RED + "Possible Overfitting Detected!" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTGREEN_EX + "No significant overfitting detected." + Style.RESET_ALL)

    return train_acc, test_acc, cross_acc, cross_std


def construct_confussion_matrix(model, X_test, y_test, label_mappings, model_name):
    """
    Function to print evaluation metrics for the model.
    It prints accuracy, weighted F1 score, confusion matrix,
    classification report, and multi-class ROC AUC score.
    """
    # Predict the results
    y_pred = model.predict(X_test)

    # Compute accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Compute weighted F1 score for multi-class
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Compute ROC AUC using one-vs-rest strategy
    try:
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)
        y_scores = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
    except Exception as e:
        roc_auc = "Not Available"

    # Print the evaluation metrics
    print(Fore.GREEN + f"\n--- {model_name} Evaluation ---\n" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Accuracy: {accuracy} | {accuracy:.4f}" + Style.RESET_ALL)
    print(Fore.LIGHTGREEN_EX + f"Weighted F1 Score: {f1} | {f1:.4f}\n" + Style.RESET_ALL)

    # Decode class labels for confusion matrix
    class_labels = [label_mappings['class'][cls] for cls in classes]

    # Print labeled confusion matrix
    print(Fore.CYAN + "Confusion Matrix (with class labels):" + Style.RESET_ALL)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(cm_df)
    print("\n")

    print(Fore.LIGHTGREEN_EX + f"Multi-Class ROC AUC (One-vs-Rest): {roc_auc}" + Style.RESET_ALL)

    # Calculate TP, TN, FP, FN for each class
    print(Fore.LIGHTGREEN_EX + "\nShowing the TP, TN, FP, FN rate for each class:" + Style.RESET_ALL)
    print('-------------------------')
    for i, class_label in enumerate(class_labels):  # Use decoded class labels
        TP = cm[i, i]  # True Positives for the current class
        FP = cm[:, i].sum() - TP  # False Positives for the current class
        FN = cm[i, :].sum() - TP  # False Negatives for the current class
        TN = cm.sum() - (TP + FP + FN)  # True Negatives for the current class

        print(Fore.YELLOW + f"Class {class_label}:" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Positives (TP): {TP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Positives (FP): {FP}" + Style.RESET_ALL)
        print(Fore.RED + f"False Negatives (FN): {FN}" + Style.RESET_ALL)
        print(Fore.GREEN + f"True Negatives (TN): {TN}" + Style.RESET_ALL)
        print('-------------------------')

    return accuracy, f1, roc_auc, cm, report


def plot_roc_auc(model, X_test, y_test):
    """
    Function to create a plot for the AUC and ROC curves.
    It takes the labels (classes), gets the probability metrics,
    and calculates the AUC ROC curve for all.
    Returns:
        matplotlib.figure.Figure: The ROC curve plot.
    """
    from matplotlib.figure import Figure

    classes = np.unique(y_test)  # Dynamically get unique classes
    y_test_bin = label_binarize(y_test, classes=classes)
    y_scores = model.predict_proba(X_test)  # Get probability scores

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for i in range(y_test_bin.shape[1]):
        if np.sum(y_test_bin[:, i]) == 0:  # Skip classes with no positive samples
            print(f"Skipping class {classes[i]} (no positive samples)")
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class ROC Curve")
    ax.legend()

    return fig

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
