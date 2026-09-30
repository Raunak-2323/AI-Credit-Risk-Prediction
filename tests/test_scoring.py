import pandas as pd, pytest
from src.scoring import pd_to_score, risk_band

def test_reference_point():
    assert pd_to_score(0.05) == 600

def test_score_bounds():
    assert pd_to_score(0.0001) <= 850
    assert pd_to_score(0.9999) == 300

def test_monotonic():
    s = [pd_to_score(p) for p in (0.01, 0.05, 0.2, 0.6, 0.95)]
    assert s == sorted(s, reverse=True)

def test_bands():
    assert risk_band(300) == "Very high risk"
    assert risk_band(580) == "High risk"
    assert risk_band(700) == "Medium risk"
    assert risk_band(760) == "Low risk"
    assert risk_band(820) == "Very low risk"

def test_model_predicts_valid_probability():
    joblib = pytest.importorskip("joblib")
    model = joblib.load("models/credit_risk_rf.joblib")
    row = pd.DataFrame([{"person_age": 30, "person_income": 55000, "person_home_ownership": "RENT",
        "person_emp_length": 4, "loan_intent": "PERSONAL", "loan_grade": "C", "loan_amnt": 10000,
        "loan_int_rate": 13.5, "loan_percent_income": 0.18, "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5}])
    assert 0 <= model.predict_proba(row)[0, 1] <= 1
