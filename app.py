import streamlit as st
import numpy as np
import pandas as pd
import joblib


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("credit_risk_model.pkl")
feature_cols = joblib.load("credit_risk_features.pkl")
model_name = joblib.load("credit_risk_model_name.pkl")


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)


# ==================================================
# TITLE
# ==================================================

st.title("💳 Credit Risk Prediction System")

st.write(
    "Enter the customer's financial information below "
    "to predict the risk of serious delinquency."
)

st.divider()


# ==================================================
# INPUT FIELDS
# ==================================================

st.subheader("Customer Financial Information")


revolving_utilization = st.number_input(
    "Revolving Utilization of Unsecured Lines",
    min_value=0.0,
    value=0.20,
    step=0.01
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

days_30_59 = st.number_input(
    "Times 30–59 Days Past Due",
    min_value=0,
    value=0
)

debt_ratio = st.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.30,
    step=0.01
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

open_credit_lines = st.number_input(
    "Open Credit Lines and Loans",
    min_value=0,
    value=5
)

days_90 = st.number_input(
    "Times 90+ Days Late",
    min_value=0,
    value=0
)

real_estate_loans = st.number_input(
    "Number of Real Estate Loans or Lines",
    min_value=0,
    value=0
)

days_60_89 = st.number_input(
    "Times 60–89 Days Past Due",
    min_value=0,
    value=0
)

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    value=0
)


# ==================================================
# PREDICT
# ==================================================

st.divider()

if st.button("🔍 Predict Credit Risk", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame([[
        revolving_utilization,
        age,
        days_30_59,
        debt_ratio,
        monthly_income,
        open_credit_lines,
        days_90,
        real_estate_loans,
        days_60_89,
        dependents
    ]], columns=feature_cols)


    # Make prediction
    prediction = model.predict(input_data)[0]

    # Probability of serious delinquency
    probability = model.predict_proba(input_data)[0][1]


    # ==================================================
    # RESULT
    # ==================================================

    st.subheader("Prediction Result")

    st.metric(
        "Credit Risk Probability",
        f"{probability * 100:.2f}%"
    )


    if prediction == 1:

        st.error(
            "HIGH CREDIT RISK\n\n"
            "The model predicts that the customer is at "
            "risk of serious delinquency."
        )

    else:

        st.success(
            "LOW CREDIT RISK\n\n"
            "The model predicts that the customer is "
            "unlikely to experience serious delinquency."
        )


    st.progress(float(probability))

    st.caption(
        f"Model used: {model_name}"
    )