

# Heart Failure Prediction Using Machine Learning

## Project Overview

This project applies machine learning to predict heart failure outcomes based on clinical records. Through a structured pipeline involving data preprocessing, model training, hyperparameter tuning, and evaluation, we compare multiple machine learning models and select the best-performing one to accurately predict heart failure risk. 

The project explores the potential of using machine learning for healthcare, specifically in predicting health outcomes, and emphasizes a workflow suited to handling medical datasets.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Custom Functions](#custom-functions)
- [Results](#results)
- [Getting Started](#getting-started)
- [Help](#help)
- [Acknowledgements](#acknowledgements)

## Dataset

The **Heart Failure Clinical Records Dataset** from the UCI Machine Learning Repository contains clinical data from patients with heart failure, including features like:

- **Age**: Patient age
- **Ejection Fraction**: Percentage of blood leaving the heart per contraction
- **Serum Creatinine**: Blood creatinine level, indicative of kidney function
- **Serum Sodium**: Blood sodium level, relevant to heart function

Each record is labeled with a binary outcome indicating whether the patient experienced heart failure. The dataset allows for a detailed analysis of clinical factors linked to heart failure risk.

- **Link to Dataset**: [Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

## Project Structure

```bash
Machine-Learning/Mini Project/
├── data/
│   └── heart_failure_clinical_records_dataset.csv  # Dataset file
├── notebooks/
│   └── MiniProject.ipynb                           # Main Jupyter Notebook with the project workflow
├── src/
│   └── custom_functions.py                         # Python file with custom functions and classes used in the project
├── README.md                                       # Project README file
└── requirements.txt                                # List of dependencies
```

- **`data/`**: Contains the dataset file.
- **`notebooks/`**: Contains the main Jupyter Notebook with the project’s data exploration, model training, and evaluation workflow.
- **`src/`**: Contains `custom_functions.py`, a Python file with custom utility functions for data processing, model evaluation, and visualization.
- **`requirements.txt`**: Lists the required dependencies for running the project.
- **`README.md`**: The README file describing the project and providing usage instructions.

## Models Implemented

The project explores and compares several machine learning models:

- **Logistic Regression**: A linear model commonly used for binary classification, providing high interpretability for medical applications.
- **K-Nearest Neighbors (KNN)**: A non-linear model that classifies data points based on their proximity to the nearest neighbors in feature space.
- **Support Vector Machine (SVM)**: A model that maximizes the margin between different classes for classification. In this case, the SVM uses the RBF kernel for better handling non-linear relationships.
- **Decision Tree Classifier**: A tree-based model that splits features into decision nodes for classification based on information gain.

After initial model comparison, hyperparameter tuning is performed on the best-performing model to further improve accuracy and reduce the risk of overfitting.

## Custom Functions

The `custom_functions.py` file contains several specialized functions and classes developed for this project to assist in data preprocessing, visualization, and model evaluation. Here is a breakdown of its key components:

1. **Data Preprocessing Functions**:
   - **Custom Min-Max Scaler**: Scales selected numerical features between a defined range (usually 0 and 1) to ensure model compatibility and improve convergence.
   - **Log Transformation**: Applies logarithmic scaling to features with skewed distributions, enhancing model performance on such data.

2. **Visualization Tools**:
   - **plot_indices_relation**: Creates scatter plots for specified features, comparing feature values against the dataset index. This is useful for identifying patterns or outliers.
   - **plot_class_distribution**: Displays the distribution of classes (i.e., heart failure outcomes) to help understand the class balance.
   - **plot_histograms_nonbinary_logarithmic**: Generates histograms for continuous features after applying log transformation to make skewed distributions more interpretable.
   - **plot_roc_curve**: Plots the ROC curve for model evaluation, showing the trade-off between sensitivity and specificity at various thresholds.

3. **Model Evaluation Functions**:
   - **calculate_metrics**: Computes evaluation metrics such as accuracy, precision, recall, F1 score, and AUC for a given model, providing a comprehensive performance summary.
   - **hyperparameter_tuning**: A function that systematically searches for the best hyperparameters (e.g., C and penalty for logistic regression) to optimize model performance.
   - **confusion_matrix_visualization**: Uses Seaborn to visualize the confusion matrix, enabling quick identification of model misclassifications.

Each function is modular and can be called independently or as part of the notebook workflow. This modularity makes it easy to apply these functions to new datasets or extend them for additional analyses.

## Results

Logistic Regression was found to be the best-performing model, achieving an accuracy of approximately **84.1%** and an AUC score of **0.84**. This model demonstrated strong generalizability on the test data, indicating that it effectively captures relationships between clinical features and heart failure outcomes.

**Key Observations**:
- **Influential Features**: Age, serum creatinine, and ejection fraction were identified as influential features, with their values strongly impacting model predictions.
- **Model Performance**: Logistic Regression’s performance was consistent across metrics, outperforming other models in accuracy and interpretability.

## Getting Started

### Prerequisites
- **Python 3.7** or later
- Libraries listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/Machine-Learning/Mini Project.git
   cd Machine-Learning/Mini Project/
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute the Notebook**:
   - Run `MiniProject.ipynb` in the `notebooks` folder to explore data, train models, and evaluate results.
   ```bash
   jupyter notebook notebooks/MiniProject.ipynb
   ```

4. **Use Custom Functions**:
   - You can utilize the `custom_functions.py` file in the `src` folder for standalone operations like data visualization, feature scaling, and model evaluation.

## Help

If you encounter issues, here are some common troubleshooting steps:

- **Import Errors**: Ensure all libraries in `requirements.txt` are correctly installed. Reinstalling specific packages can sometimes resolve import errors:
    ```bash
    pip install -U <package_name>
    ```
- **Dataset Loading Errors**: Verify the dataset is located in the `data/` folder and is named `heart_failure_clinical_records_dataset.csv`. Ensure the path in the code points to this file.
- **Notebook Crashes**: Restart the Jupyter kernel and clear outputs to resolve memory-related issues or crashes.

### FAQs
- **Can I add more features?**
   - Yes, the custom functions are modular and allow easy integration of additional features. Update the dataset accordingly and rerun preprocessing steps.
- **How can I change models or hyperparameters?**
   - You can adjust models and their parameters directly in the notebook or modify the `hyperparameter_tuning` function to automate this process.

If you need further assistance, feel free to submit an issue on GitHub.

## Acknowledgements

- **Dataset**: Davide Chicco and Giuseppe Jurman, *Heart Failure Clinical Records*, UCI Machine Learning Repository.
- **Libraries Used**: Scikit-Learn, Pandas, Seaborn, Matplotlib, Numpy
- **Sources**:
   - Mayo Clinic: [Heart Failure Overview](https://www.mayoclinic.org/diseases-conditions/heart-failure/symptoms-causes/syc-20373142)
   - Machine Learning Mastery: [ROC and AUC Metrics](https://machinelearningmastery.com/roc-curves-and-auc-in-machine-learning/)
