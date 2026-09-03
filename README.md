# AI-Credit-Risk-Prediction
# AI Credit Risk Prediction

A Machine Learning project for predicting the probability of serious credit delinquency within the next two years.

## Project Overview

Credit risk assessment is an important application of Machine Learning in the banking and financial sector. This project develops and compares multiple classification models to predict whether a borrower is likely to experience serious delinquency within two years.

The best-performing model is then integrated into an interactive Streamlit application for credit-risk prediction.

## Objective

The objectives of this project are:

- Predict serious delinquency within two years.
- Compare different Machine Learning classification models.
- Evaluate models using multiple performance metrics.
- Select the best-performing model.
- Develop an interactive credit-risk prediction application.

## Machine Learning Models

The following models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

The models were ranked primarily using F1-Score, followed by PR-AUC, Recall, ROC-AUC, and Accuracy.

## Best Performing Model

Random Forest was selected as the best-performing model based on the model evaluation.

Performance:

- F1-Score: 0.7747
- ROC-AUC: 0.8533
- PR-AUC: 0.8460

## Input Features

The application uses the following borrower information:

- Revolving Utilization
- Age
- Number of Times 30–59 Days Past Due
- Debt Ratio
- Monthly Income
- Open Credit Lines and Loans
- Number of Times 90+ Days Late
- Number of Real Estate Loans/Lines
- Number of Times 60–89 Days Past Due
- Number of Dependents

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
AI-Credit-Risk-Prediction/
│
├── app.py
├── credit_risk_model.pkl
├── credit_risk_features.pkl
├── credit_risk_model_name.pkl
├── credit_risk.ipynb
├── requirements.txt
├── runtime.txt
└── README.md
