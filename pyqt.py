import sys
import time

import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QLabel, QTabWidget,
    QHBoxLayout, QFileDialog, QMenuBar, QMenu, QTableWidget, QTableWidgetItem, QSpinBox, QTextEdit
)
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Model Trainer")
        self.setGeometry(100, 100, 800, 600)

        self.df: pd.DataFrame = pd.DataFrame()

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
        self.load_button = QPushButton("Load Model")
        self.train_button = QPushButton("Train Model")
        self.show_results_button = QPushButton("Show Results")

        button_style = """
            QPushButton {
                border: 2px solid #0078D7;
                border-radius: 10px;
                background-color: #E1F5FE;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #B3E5FC;
            }
        """

        for btn in [self.import_button, self.load_button, self.train_button, self.show_results_button]:
            btn.setFixedSize(150, 50)
            btn.setFont(QFont("Arial", 12))
            btn.setStyleSheet(button_style)
            operations_layout.addWidget(btn)

        self.import_button.clicked.connect(self.load_file)
        self.load_button.clicked.connect(self.process_data_pipeline)
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
        self.data_tab.setLayout(data_layout)

        # Add a QTableWidget to display the data
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(0)
        self.data_table.setRowCount(0)

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

        # Add a QSpinBox to allow the user to select how many rows to display
        self.row_spinbox = QSpinBox()
        self.row_spinbox.setMinimum(1)
        self.row_spinbox.setMaximum(100)
        self.row_spinbox.setValue(10)  # Default to showing 10 rows
        self.row_spinbox.valueChanged.connect(self.update_table)
        data_layout.addWidget(QLabel("Number of rows to display:"))
        data_layout.addWidget(self.row_spinbox)

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
        results_layout = QVBoxLayout()
        self.results_tab.setLayout(results_layout)

        # Charts Tab
        self.charts_tab = QWidget()
        charts_layout = QVBoxLayout()
        self.chart_canvas = MatplotlibCanvas()
        charts_layout.addWidget(self.chart_canvas)
        self.charts_tab.setLayout(charts_layout)

        # Adding tabs
        self.tabs.addTab(self.data_tab, "Data")
        self.tabs.addTab(self.logs_tab, "Logs")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.charts_tab, "Charts")

        main_layout.addWidget(self.tabs)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Set default theme
        self.set_light_theme()

    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def load_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;All Files (*.*)")
        if file_path:
            self.file_path_label.setText(f"Loaded File: {file_path}")
            self.df = self.load_data(file_path)
            self.update_table()

    def load_data(self, file_path):
        """
        Function to load the data from the CSV file.
        """
        try:
            df = pd.read_csv(file_path, na_values='?')  # Replace '?' with NaN
            self.log_message("Dataset loaded successfully!")
            return df
        except Exception as e:
            self.log_message(f"Failed to load dataset: {str(e)}")
            return None

    def process_data_pipeline(self):
        if self.df is None:
            self.log_message("No dataset found")
            return

        start_time = time.time()
        self.log_message("Starting Data processing...")

        # Call the processing functions in order
        self.df = self.remove_unwanted_columns(self.df)
        self.df = self.remove_outliers(self.df)
        self.df = self.fill_missing_values(self.df)
        #self.df, label_encoders, label_mappings = self.encode_categorical(self.df)

        # Log categorical mappings instead of printing
        self.log_message("\nCategorical Data Mappings:")
        #for col, mapping in label_mappings.items():
        #    self.log_message(f"{col}: {mapping}")

        elapsed_time = time.time() - start_time
        self.log_message(f"Data processing completed in {elapsed_time:.2f} seconds")

        self.update_table()

    def remove_unwanted_columns(self, df):
        """
        :param df:
        :return: df:
        function removes the unwanted columsn
        """
        columns_to_drop = [col for col in df.columns if
                           'measured' in col.lower()]  # Remove all columns that have to do with measured -> Mutually Exclusives
        df.drop(columns=columns_to_drop, inplace=True)
        self.log_message("Removed unwanted columns.")
        return df

    def remove_outliers(self, df):
        """
                Function to remove the outliers all numeric columns, namely there are some values in age that reach the 65,000 mark
        """
        self.log_message("Removing Outliers")

        numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
        Q1 = df[numerical_columns].quantile(0.25)
        Q3 = df[numerical_columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_no_outliers = df[
            ~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]

        self.log_message(f"Outliers removed: {len(df) - len(df_no_outliers)} rows dropped.")
        return df_no_outliers

    def fill_missing_values(self, df):
        """
                Function to fill all the missing values using the mean of set columnn (numeric)
        """

        self.log_message("Filling Missing Values")

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

        self.log_message("Missing Values filled")

    def encode_categorical(self, df):
        """
        Function to encode categorical data using label encoding.
        This is done so that the model can understand the categories.
        It also stores the mapping for decoding labels later.
        """

        print("\nEncoding categorical variables...")

        label_encoders = {}
        label_mappings = {}

        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            label_mappings[col] = dict(enumerate(le.classes_))  # Store mapping

        print("Categorical columns encoded.\n")

        return df, label_encoders, label_mappings

    ###### ALL THE BELOW FUNCTIONS NEED TO BE MANAGED
    ###### BY A GENERAL FUNCTION THAT WILL BE CALLED
    ###### FROM THE LOAD BUTTON

    # TODO: Implement fill missing values

    # TODO: Implement encode categorical data

    # TODO: Implement dataset balancing

    # TODO: Implement feature selection

    ###### THE TRAIN MODEL BUTTON WILL CALL THE BELOW FUNCTIONS
    ###### THESE ALSO NEED TO BE MANAGED BY A FUNCTION

    # TODO: Implement train model function -> results are stored in the logs

    # these need to be set with global variables as their results will
    # be shown in the results tab

    # TODO: Implement check overfit

    # TODO: Implement decode_predictions

    # TODO: Implement print_class_mapping

    ###### THE BELOW FUNCTIONS NEED TO BE CALLED UPON PRESSING
    ###### A NEW BUTTON TO PERFORM DATA ANALYSIS

    # TODO: plot_roc_auc, construct_confussion_matrix


    def update_table(self):
        """
        Update the table widget with the loaded data.
        """
        if hasattr(self, 'df') and self.df is not None:
            num_rows = self.row_spinbox.value()
            self.data_table.setRowCount(num_rows)
            self.data_table.setColumnCount(len(self.df.columns))
            self.data_table.setHorizontalHeaderLabels(self.df.columns)

            for i in range(num_rows):
                for j in range(len(self.df.columns)):
                    item = QTableWidgetItem(str(self.df.iloc[i, j]))
                    self.data_table.setItem(i, j, item)

    def set_light_theme(self):
        self.setStyleSheet("""
            background-color: white;
            color: black;
        """)
        self.file_path_label.setStyleSheet("border: 1px solid black;")

    def set_dark_theme(self):
        self.setStyleSheet("""
            background-color: #2E2E2E;
            color: white;
        """)
        self.file_path_label.setStyleSheet("border: 1px solid white;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())