# Dataset Exploration and Notes for Model Training

In this folder, I first go through the dataset in order to take note of some important considerations when training the model.

> **Note:**  
> The dataset used in this step is not the final dataset for training the models. I will start with a newly imported dataset and, based on the observations from this initial exploration, I will apply necessary changes and modifications.

---

### Key Takeaways from the Dataset Exploration:

- **Outlier Detection and Handling:**  
  It is crucial to **perform outlier detection and handling**, especially due to the extreme values found in the **age** column. These outliers may significantly impact model performance, and addressing them will improve the results.

- **Feature Selection:**  
  **Feature selection** is necessary because some columns are **mutually exclusive**. Removing redundant features will simplify the model and improve its generalization ability.

- **Class Transformation:**  
  The classes in the dataset currently include **both the class label and an identification number**. These identification numbers must be removed, and the class should be transformed into a more general and meaningful label. This will ensure the classes are consistent and ready for analysis.
