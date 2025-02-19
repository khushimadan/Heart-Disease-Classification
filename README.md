# Heart Disease Prediction Using Machine Learning

## Description

This project applies machine learning to predict heart disease based on clinical attributes. It explores various models, including Logistic Regression, K-Nearest Neighbors, and Random Forest, to determine the most accurate prediction method.

## Dataset

Source: UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/45/heart+disease

Alternative: Kaggle Dataset - https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

The dataset includes features such as age, chest pain type, blood pressure, cholesterol levels, and more.

## Approach

Data Exploration & Preprocessing: Understanding distributions, handling missing values, and feature selection.

Model Training & Evaluation: Comparing machine learning models using metrics like accuracy, precision, and recall.

Hyperparameter Tuning: Optimizing models using GridSearchCV & RandomizedSearchCV.

Visualization: Using Seaborn & Matplotlib to analyze key relationships in the data.

## Installation & Usage

To run this project, follow these steps:

git clone https://github.com/khushimadan/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
pip install -r requirements.txt
jupyter notebook

Then open Heart-Disease-Classification.ipynb.

## Dependencies

Python

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn (for model training & evaluation)

Jupyter Notebook

## Results

Achieved an accuracy of ~90%

The best-performing model was Logistic Regression

## Future Improvements

Try deep learning (e.g., Neural Networks).

Expand dataset with more medical parameters.

Deploy model using Flask or FastAPI.
