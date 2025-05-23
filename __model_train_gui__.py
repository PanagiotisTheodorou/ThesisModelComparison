
"""
    This file will not contain many comments, since the code for pyqt illustrates the functionality really well
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QTabWidget,
    QHBoxLayout, QFileDialog, QMenuBar, QMenu, QTableWidget, QTableWidgetItem, QSpinBox,
    QTextEdit, QProgressDialog, QDialog, QScrollArea, QFrame, QSizePolicy, QComboBox
)
from PyQt6.QtGui import QFont, QAction, QMovie
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import pandas as pd
from colorama import Fore, Style

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Global dataframe
df = pd.DataFrame()

from __pyqt_charts__ import plot_prediction_distribution, plot_precision_recall_curve, plot_feature_importance, construct_confusion_matrix_visual, plot_roc_auc

# Import utility functions
from __general_utils__ import (
    remove_unwanted_columns,
    remove_outliers,
    fill_missing_values,
    encode_categorical
)

from ModelsUsedByUI.train_RFC import train_RFC

from ModelsUsedByUI.train_SVM import train_SVM

from ModelsUsedByUI.train_DT import train_DT

from ModelsUsedByUI.train_KNN import train_KNN

from ModelsUsedByUI.train_NV import train_NB

from ModelsUsedByUI.train_LogR import train_LogR

from __postprocessing_utils__ import construct_confussion_matrix_logical, check_overfitting

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class LoadingWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Processing")
        self.setFixedSize(250, 250) 
        self.setWindowModality(Qt.WindowModality.ApplicationModal)  # Block interactions with the main UI

        layout = QVBoxLayout()

        # QLabel for the anmation
        self.loading_label = QLabel(self)
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Load animation
        self.loading_movie = QMovie("load_indicator.gif")
        self.loading_label.setMovie(self.loading_movie)
        self.loading_movie.start()  # Start animation

        self.message_label = QLabel("Building the Machine Learning model\nPlease wait.", self)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # append the widhets
        layout.addWidget(self.loading_label)
        layout.addWidget(self.message_label)

        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Trainer")
        self.setGeometry(100, 100, 800, 600)

        # Copy global df
        self.df: pd.DataFrame = df.copy()

        self.model_options = {
            "Random Forest": train_RFC,
            "Support Vector Machine (SVM)": train_SVM,
            "Linear Regression": train_LogR,
            "Decision Tree": train_DT,
            "KNN": train_KNN,
            "Naive Bayes": train_NB
        }

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

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_application_state)
        self.reset_button.setFixedSize(150, 50)
        self.reset_button.setFont(QFont("Arial", 12))
        self.reset_button.setStyleSheet("border: 2px solid #FF0000; border-radius: 10px; background-color: #FFCCCB;")
        operations_layout.addWidget(self.reset_button)

        main_layout.addLayout(operations_layout)

        self.file_path_label = QLabel("No file loaded.")
        self.file_path_label.setStyleSheet("font-style: italic; text-decoration: underline;")
        main_layout.addWidget(self.file_path_label)

        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)

        line_top = QFrame()
        line_top.setFrameShape(QFrame.Shape.HLine)
        line_top.setFrameShadow(QFrame.Shadow.Sunken)
        line_top.setLineWidth(1)
        model_layout.addWidget(line_top)

        model_label = QLabel("Select Model:")
        model_layout.addWidget(model_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(self.model_options.keys())
        self.model_dropdown.setStyleSheet("""
            QComboBox {
                border: 2px solid #0078D7;
                border-radius: 5px;
                padding: 5px;
                background-color: #E1F5FE;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        model_layout.addWidget(self.model_dropdown)

        # Bottom dividing point
        line_bottom = QFrame()
        line_bottom.setFrameShape(QFrame.Shape.HLine)
        line_bottom.setFrameShadow(QFrame.Shadow.Sunken)
        line_bottom.setLineWidth(1)
        model_layout.addWidget(line_bottom)

        main_layout.addWidget(model_widget)

        # Tab for Data, Logs,and  Results
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { /* The tab widget frame */
                border: 2px solid #0078D7;
                top: -1px;
            }

            QTabBar::tab {
                background: #E1F5FE;
                border: 1px solid #0078D7;
                padding: 5px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 80px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid white;
            }

            QTabBar::tab:hover {
                background: #d0e7ff;
            }
        """)

        # Data Tab
        self.data_tab = QWidget()
        data_layout = QVBoxLayout()
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)
        self.data_table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)
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
        self.data_tab.setLayout(data_layout)

        # Logs Tab
        self.logs_tab = QWidget()
        logs_layout = QVBoxLayout()
        self.logs_tab.setLayout(logs_layout)

        # QTextEdit for logging messages
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        logs_layout.addWidget(self.log_text_edit)

        # Results Tab
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_tab.setLayout(self.results_layout)

        #handle overflow
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content) 
        self.results_layout.addWidget(self.scroll_area) 

        # allocate tabs
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.results_tab, "Results")

        main_layout.addWidget(self.tabs)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.set_light_theme()

    def reset_application_state(self):
        """
            Resets the application state by re-importing the dataset and clearing logs and charts.
         """

        self.log_text_edit.clear()

        # Clear charts
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        # reimport the dataset
        if hasattr(self, 'file_path'):
            global df
            df = self.load_data(self.file_path)
            self.df = df.copy()
            self.display_data()
            self.log_message("Dataset re-imported successfully.")
        else:
            self.log_message("No dataset loaded. Please import a dataset first.")

        self.log_message("Application state reset successfully.")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def print_message(self, message, color):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = getattr(Fore, color.upper(), Fore.WHITE)
        print(color_code + f"[{timestamp}]\n{message}\n" + Style.RESET_ALL)

    def print_message_extended(self, message, color):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color_code = getattr(Fore, color.upper(), Fore.WHITE)
        print(color_code + f"[{timestamp}]\n{message}" + Style.RESET_ALL)

    # Functions that regard the functionality of the model

    def import_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            global df
            self.file_path = file_path
            df = self.load_data(file_path)
            self.df = df.copy()
            self.file_path_label.setText(f"Loaded: {file_path}")
            self.display_data()

    def load_data(self, file_path):
        self.log_message("Loading dataset")
        self.print_message("Loading dataset", "GREEN")
        df = pd.read_csv(file_path, na_values='?')
        return df

    def display_data(self):
        if self.df.empty:
            return

        progress = QProgressDialog("Loading data", None, 0, len(self.df), self)
        progress.setWindowTitle("Please wait")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(500) 

        self.data_table.setColumnCount(len(self.df.columns))
        self.data_table.setRowCount(len(self.df))
        self.data_table.setHorizontalHeaderLabels(self.df.columns)

        for i, row in self.df.iterrows():
            progress.setValue(i)
            QApplication.processEvents()
            for j, cell in enumerate(row):
                self.data_table.setItem(i, j, QTableWidgetItem(str(cell)))

        progress.setValue(len(self.df))

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

        fig.tight_layout(pad=2.0)

        # embed the figure in a canvas
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # add frame fr the chart
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.Shape.Box)
        chart_frame.setMinimumSize(400, 600)

        # add spacing for clarity
        frame_layout = QVBoxLayout(chart_frame)
        frame_layout.addWidget(canvas)
        frame_layout.addSpacing(20)

        self.scroll_layout.addWidget(chart_frame)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setLineWidth(1)
        self.scroll_layout.addWidget(divider)

        canvas.draw()
        self.scroll_content.adjustSize()


    def perform_processing(self):
        # Reset the application state before training a new model
        self.reset_application_state()

        self.log_message("Initializing data processing")

        # Show the loading window
        self.loading_window = LoadingWindow()
        self.loading_window.show()

        # Ensure the UI update
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

            # Step 4: Enode categorical variables
            self.log_message("Encoding categorical variables")
            self.df, self.label_encoders, self.label_mappings = encode_categorical(df)
            self.log_message("Categorical columns encoded.")

            # Get the selected model from the dropdown
            selected_model_name = self.model_dropdown.currentText()
            selected_model_func = self.model_options[selected_model_name]

            self.log_message(f"Training {selected_model_name} model")
            model, X_train, X_test, y_train, y_test, acr, best_params, cr = selected_model_func(
                df, target_column, self.label_mappings
            )

            self.log_message("Starting model training")
            self.log_message("Filtering rare classes")
            self.log_message("Applying dataset balancing")
            self.log_message("Dataset balanced.")
            self.log_message("Performing stratified train-test split")
            self.log_message("Hyperparameter tuning using GridSearchCV")

            self.log_message(f"Accuracy: {acr:.4f}")
            self.log_message(f"Best Parameters: {str(best_params)}")

            self.log_message("Checking for Overfitting")
            train_acc, test_acc, cross_acc, cross_std = check_overfitting(model, X_train, y_train, X_test, y_test)
            self.log_message(f"Training Accuracy: {train_acc:.4f}")
            self.log_message(f"Testing Accuracy: {test_acc:.4f}")
            self.log_message(f"Cross-Validation Accuracy: {cross_acc:.4f} +/- {cross_std:.4f}")

            if train_acc > test_acc + 0.05:
                self.log_message("Possible Overfitting Detected!")
            else:
                self.log_message("No significant overfitting detected.")

            construct_confussion_matrix_logical(model, X_test, y_test, self.label_mappings, model_name="Random Forest")

            self.log_message("Generating Confusion Matrix")
            cm_fig = construct_confusion_matrix_visual(model, X_test, y_test, self.label_mappings, model_name="Random Forest")
            self.add_chart_to_results_tab(cm_fig, row=0, col=0)

            self.log_message("Generating ROC Curve")
            roc_fig = plot_roc_auc(model, X_test, y_test)
            self.add_chart_to_results_tab(roc_fig, row=1, col=1)

            self.log_message("Generating Feature Importance Plot")
            feature_names = X_train.columns.tolist()
            feature_importance_fig = plot_feature_importance(model, feature_names)
            self.add_chart_to_results_tab(feature_importance_fig, row=2, col=2)

            self.log_message("Generating Precision-Recall Curve")
            pr_curve_fig = plot_precision_recall_curve(model, X_test, y_test)
            self.add_chart_to_results_tab(pr_curve_fig, row=3, col=3)

            self.log_message("Generating Prediction Distribution Plot")
            y_pred = model.predict(X_test)
            pred_dist_fig = plot_prediction_distribution(y_test, y_pred, self.label_mappings)
            self.add_chart_to_results_tab(pred_dist_fig, row=4, col=4)

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())