Resample and Model Evaluation

Overview

This project explores different resampling techniques and machine learning models for credit risk evaluation. The goal is to handle class imbalance in the dataset effectively and determine the best model using evaluation metrics such as F2-score, ROC-AUC, precision, and recall.

Features
Data Preprocessing: Cleaning and encoding categorical variables.
Resampling Techniques: Includes Random Oversampling, SMOTE, and Random Undersampling.
Machine Learning Models: Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, and others.
Model Evaluation: Uses cross-validation, precision-recall curves, and ROC analysis.
Hyperparameter Tuning: Fine-tunes Gradient Boosting using GridSearchCV.

Installation

Usage
Ensure you have the dataset (credit_risk_for_exam.zip) in the correct directory. The script performs:

Data preprocessing
Resampling to balance the dataset
Model training and evaluation
Hyperparameter tuning for Gradient Boosting
Run the script:


Dependencies
Python (>=3.7)
pandas
numpy
scikit-learn
imbalanced-learn
scipy
seaborn
matplotlib

Install them using:
pip install pandas numpy scikit-learn imbalanced-learn scipy seaborn matplotlib

Results
The best resampling method and model are determined based on F2-score and AUC. The final results are exported to an Excel file for further analysis.

Notes
The dataset path may need to be adjusted in the script.
Some print statements and commented-out code remain for debugging and exploration.
