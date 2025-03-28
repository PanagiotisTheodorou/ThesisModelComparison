# Data Directory Overview

## Contents
This directory contains all datasets used in the system, including raw data, preprocessed versions, and data analysis outputs.

## Key Files

### Primary Datasets
- `001_dataset_before_preprocessing.csv`  
  Original raw dataset before any transformations
- `002_dataset_after_filling_missing_values.csv`  
  THis dataset is the result of filling the orinal dataset, where it has missing values
- `003_dataset_after_removing_outliers.csv`  
  This dataset is the state of the original, once missing values are filled, and outliers removed
- `raw.csv`  
  Preprocessed dataset, that can be used for model training, but has all the classes
- `raw_with_general_classes.csv`  
  Processed dataset actually used for model training, this one has the altered classes

### Supporting Files
- `cleaned_dataset_after_dt.csv`  
  output form the decision tree model training
- `cleaned_dataset_after_knn.csv`  
  output form the knn model training
- `cleaned_dataset_after_linr.csv`  
  output form the linear regression model training
- `cleaned_dataset_after_logr.csv`  
  output form the logistic regression model training
- `cleaned_dataset_after_nb.csv`  
  output form the naive bayes model training
- `cleaned_dataset_after_rf.csv`  
  output form the random forest model training
- `cleaned_dataset_after_svm.csv`  
  output form the svm model training