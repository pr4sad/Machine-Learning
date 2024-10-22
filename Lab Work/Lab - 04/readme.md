# Support Vector Machine (SVM)

This repository demonstrates the application of the **Support Vector Machine (SVM)** algorithm to classify patients with and without Parkinson's disease based on voice measurements.


## Overview

- **SVM Algorithm**: Used for binary classification to distinguish between Parkinson's disease patients and healthy individuals.
- **Dataset**: The **Parkinson's dataset** contains voice measurement features which are key indicators of the disease.
- **Objective**: Build a model that can accurately classify whether a person has Parkinson’s disease based on their vocal metrics.

## Workflow

1. **Data Preprocessing**:
   - Handle missing values and clean the dataset (if necessary).
   - Normalize the features for better SVM performance.

2. **Feature Selection**:
   - Utilize key features such as pitch, jitter, shimmer, and harmonic-to-noise ratio.

3. **Train-Test Split**:
   - Split the data into training and testing sets for model evaluation.

4. **Model Training**:
   - Train the **Support Vector Machine (SVM)** using different kernel functions (linear, polynomial, RBF).
   
5. **Model Evaluation**:
   - Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrix.

## Dataset

The **Parkinson's dataset** contains 195 voice recordings from 31 individuals, including:
- **Features**: 22 voice metrics such as `MDVP:Fo(Hz)`, `MDVP:Jitter(%)`, `MDVP:Shimmer(dB)`, `NHR`, `HNR`, etc.
- **Target Variable**: Binary target indicating whether the individual has Parkinson's disease (`1`) or is healthy (`0`).

## Key Features
- **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
- **MDVP:Jitter(%)**: Variation in frequency.
- **MDVP:Shimmer(dB)**: Variation in amplitude.
- **NHR**: Noise-to-harmonics ratio, a measure of noise in the voice.
- **HNR**: Harmonics-to-noise ratio, indicating voice quality.

## Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
