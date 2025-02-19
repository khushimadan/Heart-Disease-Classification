# Heart Disease Prediction Using Machine Learning

## Description

This project applies machine learning to predict heart disease based on clinical attributes. It explores various models, including Logistic Regression, K-Nearest Neighbors, and Random Forest, to determine the most accurate prediction method.

## Dataset

Source: UCI Machine Learning Repository

Alternative: Kaggle Dataset

The dataset includes features such as age, chest pain type, blood pressure, cholesterol levels, and more.

## Approach

Data Exploration & Preprocessing: Understanding distributions, handling missing values, and feature selection.

Model Training & Evaluation: Comparing machine learning models using metrics like accuracy, precision, and recall.

Hyperparameter Tuning: Optimizing models using GridSearchCV & RandomizedSearchCV.

Visualization: Using Seaborn & Matplotlib to analyze key relationships in the data.

## Installation & Usage

To run this project, follow these steps:

git clone https://github.com/your_username/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
jupyter notebook

Then open Heart-Disease-Classification.ipynb.

## Dependencies

Python

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn (for model training & evaluation)

Jupyter Notebook

## Results

Achieved a target accuracy of ~95% (update based on actual results).

The best-performing model was Random Forest (modify if needed).

## Future Improvements

Try deep learning (e.g., Neural Networks).

Expand dataset with more medical parameters.

Deploy model using Flask or FastAPI.
