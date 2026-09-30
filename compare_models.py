"""Compare 3 algorithms on 5 metrics. Usage: python compare_models.py"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.pipeline import load_data, preprocessor

X, y = load_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
algos = {
    "Logistic Regression": (True, LogisticRegression(max_iter=1000)),
    "Random Forest": (False, RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)),
    "Gradient Boosting": (False, GradientBoostingClassifier(random_state=42)),
}
rows = []
for name, (scale, clf) in algos.items():
    m = Pipeline([("prep", preprocessor(X, scale)), ("clf", clf)]).fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred), "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred), "F1": f1_score(y_te, pred), "ROC-AUC": roc_auc_score(y_te, p)})
res = pd.DataFrame(rows).set_index("Model").round(4)
print(res); res.to_csv("reports/model_comparison.csv")
