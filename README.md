# SETA – Use Structured and Explainable Thyroid Detection System Using AI

**Panagiotis Theodorou**  
[PTheodorou@uclan.ac.uk](mailto:PTheodorou@uclan.ac.uk)  
[GitHub](https://github.com/PanagiotisTheodorou) | [LinkedIn](https://www.linkedin.com/in/panagiotis-theodorou-a16519303/)

**Git URL**
https://github.com/PanagiotisTheodorou/ThesisModelComparison

<!-- TABLE OF CONTENTS -->
<details>
  <summary><h2>Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#structure">Project Structure</a></li>
      </ul>
    </li>
    <li>
      <a href="#development-tools">Development Tools</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

The project falls into the category of Research and Development. It aims as fixing key issues that are currently present in the use of ML models for medical purposes. Specifically, it provides a structured approach towards training refined ML models that predict thyroid disease. These models are supplemented by the use of two graphical user interfaces, one for Machine Learning experts, data analysts, and researchers, and another for doctors, and practitioners. Thus, this project aims to bridge the gap between ML and Healthcare, by creating a highly accessible system, that also provides in terms of capabilities.

In depth view of specifications: 

- **Dual-Interface System**:
  - **Training Interface (PyQt)**: For ML experts to develop and analyze thyroid prediction models
  - **Inference Interface (Streamlit)**: For medical practitioners to make clinical predictions

- **Key Features**:
  - Comprehensive model comparison (KNN, SVM, Random Forest, etc.)
  - Automated data preprocessing pipeline
  - Interactive visualization tools
  - Medical-grade prediction interface

- **Target Users**:
  - Medical researchers and data scientists
  - Healthcare practitioners
  - Clinical decision support specialists

### Project Structure

*Each folder contains detailed markdown documentation explaining its contents and purpose.*

- 000_Data, this folder holds the data that is either used or generated by scripts.

- 001_PreProcessing, this folder has scripts that analyze and perform preprocessing steps, for documentation pusposes. The findings and procedures used in this section, are to be used later on in the model training phase. This section gives more emphasis on the structure and validity of the dataset.

- 002_DataAnalysis, this folder expands on what the previous had performed, while going a bit more in detail on the theoretical aspects of the dataset.

- 003_ModelComparison, this folder uses the findings from folders 002 and 003 to train models and compare them

- 005_UserUI, this folder holds a streamlit application that acts as the inference point, which the healthcae individuals will be using. In this folder, the previously created pickle files are stored, and accessed by the streamlit application.

- In the root of the application (not placed in a folder), lies a pyqt application, which is the training interface for the ML experts. It allows the users to train ML models, while viewing logs, analyzing data and viewing charts.

## Development Tools

### Core Technologies
- Python 3.9+
- PyQt6 (Training Interface)
- Streamlit (Inference Interface)
- scikit-learn
- pandas
- matplotlib/seaborn

### Machine Learning Models
- Random Forest
- Support Vector Machines (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Naive Bayes
- Linear Regression


## Getting Started

### Clone the repository
```bash
git clone https://github.com/PanagiotisTheodorou/ThesisModelComparison.git
```
```bash
cd ThesisModelComparison
```

### Setup Virtual Environment 

For anaconda users
```bash
conda create -n seta_env python=3.9
```
```bash
conda activate seta_env
```

For standard python:
```bash
python -m venv venv
```
Linux/Mac
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Usage

To run the training interface, visit the "__model_train_gui__.py" and press run, if you are using pycharm,
otherwise, use:
```bash
python __model_train_gui__py
```

To run the inference interface, use:
```bash
cd 005_User_UI
```
```bash
streamlit run streamlitApp.py
```

## Acknowledgemernts

Doctor Andria Procopiou - UCLan Cyprus