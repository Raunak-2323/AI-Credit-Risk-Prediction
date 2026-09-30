"""Train the Random Forest, evaluate it, and save model + metrics + scored test set.
Usage: python train.py"""
import json, joblib, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from src.pipeline import load_data, build_random_forest
from src.scoring import pd_to_score, risk_band

THRESHOLD = 0.5   # raise for higher precision, lower for higher recall

X, y = load_data()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = build_random_forest(X_tr).fit(X_tr, y_tr)

prob = model.predict_proba(X_te)[:, 1]
pred = (prob >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
metrics = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred),
           "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred),
           "roc_auc": roc_auc_score(y_te, prob), "threshold": THRESHOLD,
           "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
           "train_rows": len(X_tr), "test_rows": len(X_te)}
metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
for k, v in metrics.items(): print(f"{k:17s}: {v}")

joblib.dump(model, "models/credit_risk_rf.joblib", compress=3)
json.dump(metrics, open("models/metrics.json", "w"), indent=2)

out = X_te.copy()
out["actual_default"] = y_te.values
out["prob_default"] = prob.round(4)
out["credit_score"] = pd_to_score(prob)
out["risk_band"] = out["credit_score"].map(risk_band)
out.to_csv("reports/scored_test_set.csv", index=False)
print("\nDefault rate by risk band:")
print(out.groupby("risk_band")["actual_default"].agg(["count", "mean"]).round(3))
