"""Data loading and the Random Forest pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "loan_status"   # 1 = default

def load_data(path="data/credit_risk_dataset.csv"):
    df = pd.read_csv(path)
    df = df[(df.person_age <= 100) & (df.person_emp_length.fillna(0) <= 60)]  # impossible outliers
    return df.drop(columns=TARGET), df[TARGET]

def preprocessor(X, scale=False):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    steps = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(steps), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

def build_random_forest(X):
    return Pipeline([("prep", preprocessor(X)),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                   n_jobs=-1, random_state=42))])
