# AI Credit Risk Prediction & Scoring

A Machine Learning project for predicting the probability that a loan applicant will default, and converting it into a 300-850 credit score.

## Project Overview

Credit risk assessment is an important application of Machine Learning in the banking and financial sector. This project develops and compares multiple classification models to predict whether a borrower is likely to default on a loan.

The best-performing model is integrated into an interactive Streamlit application for credit-risk prediction and scoring.

## Objective

* Predict loan default.
* Compare different Machine Learning classification models.
* Evaluate models using multiple performance metrics.
* Select the best-performing model.
* Convert default probability into a 300-850 credit score and risk band.
* Develop an interactive credit-risk prediction application.

## Dataset

`data/credit_risk_dataset.csv` contains 32,581 loans, of which about 22% defaulted. The target is `loan_status` (1 = default). Impossible values (age over 100, employment length over 60 years) are removed before training.

## Machine Learning Models

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

## Results (20% hold-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 86.4% | 76.9% | 53.8% | 0.633 | 0.862 |
| **Random Forest** | **93.0%** | **97.9%** | 69.2% | **0.811** | **0.926** |
| Gradient Boosting | 92.3% | 94.4% | 68.8% | 0.796 | 0.921 |

## Best Performing Model

Random Forest was selected as the best-performing model, with the highest accuracy, precision, F1-Score and ROC-AUC.

Performance:

* Accuracy: 0.9295
* Precision: 0.9791
* Recall: 0.6918
* F1-Score: 0.8107
* ROC-AUC: 0.9259

## Credit Score

The default probability is converted into a 300-850 score: 600 points at 5% default probability, and +50 points each time the odds of repaying double.

| Score | Risk band | Default rate in test set |
|---|---|---|
| Below 580 | Very high risk | 36.5% |
| 580-669 | High risk | 3.8% |
| 670-739 | Medium risk | 0.6% |
| 740-799 | Low risk | 0.0% |
| 800+ | Very low risk | 0.0% |

## Input Features

* Age
* Annual Income
* Home Ownership
* Employment Length (years)
* Loan Purpose
* Loan Grade
* Loan Amount
* Interest Rate
* Loan as Share of Income
* Earlier Default on File
* Credit History Length (years)

## Project Structure

```
├── app.py                  Streamlit web app
├── train.py                Trains the Random Forest, saves model and metrics
├── compare_models.py       Compares the 3 algorithms
├── src/pipeline.py         Data loading, preprocessing, model
├── src/scoring.py          Probability to score to risk band
├── tests/test_scoring.py   Unit tests
├── data/                   Dataset
├── models/                 Saved model and metrics
├── reports/                Comparison and scored test set
├── requirements.txt
└── runtime.txt
```

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python compare_models.py
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud with Python 3.12. `scikit-learn` is pinned to the version used to train the saved model.

## Limitations

The dataset is a public practice dataset (median borrower age about 26). This project is for learning and demonstration. A real lending system would need its own data, validation, fairness testing and regulatory review.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook
