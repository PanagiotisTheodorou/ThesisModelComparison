# SETA

## User Structured Thyroid Prediction Application

Panagiotis Theodorou  
<a href="mailto:your.email@example.com">ptheodorou@uclan.ac.uk</a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary><h2>Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#components">System Components</a></li>
        <li><a href="#tools">Development Tools</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#trainer-app">Model Trainer (PyQt)</a></li>
        <li><a href="#inference-app">Inference App (Streamlit)</a></li>
      </ul>
    </li>
    <li><a href="#structure">Project Structure</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

This project provides a comprehensive solution for thyroid disease classification with two specialized applications:

1. **Model Trainer Application (PyQt)**: A desktop GUI application for training and evaluating multiple machine learning models on thyroid disease data.
2. **Inference Application (Streamlit)**: A web-based interface for end-users to make predictions using pre-trained models.

Key Features:
- Compare performance of 7 different ML algorithms
- Comprehensive data preprocessing pipeline
- Model evaluation with multiple metrics
- User-friendly prediction interface
- Detailed data visualizations

Intended Users:
- Medical researchers developing thyroid disease models
- Healthcare professionals needing diagnostic assistance
- Data scientists working with medical datasets


### Development Tools

* **Core Technologies**:
  - Python 3.9+
  - PyQt6 (for Trainer App)
  - Streamlit (for Inference App)
  - scikit-learn
  - pandas
  - matplotlib

* **Machine Learning**:
  - Random Forest
  - SVM
  - Logistic Regression
  - Decision Trees
  - KNN
  - Naive Bayes
  - Linear Regression

* **Utilities**:
  - joblib (model serialization)
  - colorama (console coloring)
  - seaborn (visualizations)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- For PyQt app (Linux users may need additional dependencies):
  ```sh
  # Ubuntu/Debian
  sudo apt-get install python3-pyqt6