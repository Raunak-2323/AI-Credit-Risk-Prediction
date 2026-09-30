import json, joblib, pandas as pd, streamlit as st
from src.scoring import pd_to_score, risk_band as band

st.set_page_config(page_title="Credit Risk Scorer", page_icon="💳", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/credit_risk_rf.joblib")

def dial(score=None, label=""):
    angle = -90 if score is None else -90 + (score - 300) / 550 * 180
    col = "#17222b" if score is None else ("#b3261e" if score < 580 else "#b7791f" if score < 740 else "#2f7d4f")
    num = "" if score is None else score
    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif">
<svg viewBox="0 0 340 268" style="width:100%;max-width:360px;display:block;margin:auto">
<path d="M20 170A150 150 0 0 1 174.2 20.1" fill="none" stroke="#b3261e" stroke-width="18"/>
<path d="M174.2 20.1A150 150 0 0 1 291.4 81.8" fill="none" stroke="#b7791f" stroke-width="18"/>
<path d="M291.4 81.8A150 150 0 0 1 320 170" fill="none" stroke="#2f7d4f" stroke-width="18"/>
<text x="20" y="192" font-size="11" fill="#5d6b76" text-anchor="middle">300</text>
<text x="320" y="192" font-size="11" fill="#5d6b76" text-anchor="middle">850</text>
<g style="transform-origin:170px 170px;transform:rotate({angle}deg);transition:transform .9s">
<line x1="170" y1="170" x2="170" y2="44" stroke="#17222b" stroke-width="3" stroke-linecap="round"/></g>
<circle cx="170" cy="170" r="9" fill="#17222b"/>
<text x="170" y="236" font-size="54" font-weight="700" text-anchor="middle" fill="#17222b">{num}</text>
<text x="170" y="260" font-size="15" font-weight="600" text-anchor="middle" fill="{col}">{label}</text>
</svg></div>"""

model = load_model()
st.title("Credit risk scorer")
st.caption("Random Forest model. Enter the applicant and loan details to get a default probability, a 300-850 score and a decision.")

left, right = st.columns([1.15, 1], gap="large")

with left:
    with st.form("applicant"):
        st.subheader("Applicant")
        a1, a2 = st.columns(2)
        age = a1.number_input("Age", 18, 100, 30)
        income = a2.number_input("Annual income", 1000, 6_000_000, 55000, step=1000)
        home = a1.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        emp = a2.number_input("Years employed", 0.0, 60.0, 4.0, step=0.5)
        hist = a1.number_input("Credit history (years)", 0, 60, 5)
        cb_def = a2.selectbox("Earlier default on file", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
        st.subheader("Loan")
        b1, b2 = st.columns(2)
        intent = b1.selectbox("Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        grade = b2.selectbox("Loan grade", list("ABCDEFG"), index=2)
        amount = b1.number_input("Loan amount", 500, 35000, 10000, step=500)
        rate = b2.number_input("Interest rate (%)", 1.0, 30.0, 13.5, step=0.1)
        threshold = st.slider("Reject when default probability is at least", 0.1, 0.9, 0.5, 0.05, format="%.2f")
        go = st.form_submit_button("Score applicant", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    if go:
        row = pd.DataFrame([{
            "person_age": age, "person_income": income, "person_home_ownership": home,
            "person_emp_length": emp, "loan_intent": intent, "loan_grade": grade,
            "loan_amnt": amount, "loan_int_rate": rate,
            "loan_percent_income": round(amount / income, 2),
            "cb_person_default_on_file": cb_def, "cb_person_cred_hist_length": hist}])
        p = float(model.predict_proba(row)[0, 1]); s = pd_to_score(p)
        st.iframe(dial(s, band(s)), height=290)
        if p >= threshold: st.error("Reject. Default probability is at or above your threshold.")
        else: st.success("Approve. Default probability is below your threshold.")
        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{p:.1%}")
        m2.metric("Loan as share of income", f"{amount / income:.0%}")
    else:
        st.iframe(dial(), height=290)
        st.info("Fill in the form and choose Score applicant.")

with st.sidebar:
    st.subheader("Model performance")
    st.caption("Random Forest on a 20% held-out test set")
    try:
        m = json.load(open("models/metrics.json"))
        st.metric("Accuracy", f"{m['accuracy']:.1%}")
        st.metric("Precision", f"{m['precision']:.1%}")
        st.metric("Recall", f"{m['recall']:.1%}")
        st.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    except FileNotFoundError:
        st.info("Run train.py to generate metrics.")
