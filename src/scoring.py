"""Convert default probability into a 300-850 credit score and a risk band."""
import numpy as np

SCORE_MIN, SCORE_MAX = 300, 850
BANDS = [(580, "Very high risk"), (670, "High risk"), (740, "Medium risk"), (800, "Low risk")]

def pd_to_score(p, base=600, pdo=50, odds0=1 / 19):
    """600 points at 5% default probability; +50 points each time good:bad odds double."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    factor = pdo / np.log(2)
    offset = base - factor * np.log(1 / odds0)
    score = np.clip(offset + factor * np.log((1 - p) / p), SCORE_MIN, SCORE_MAX).round().astype(int)
    return int(score) if score.ndim == 0 else score

def risk_band(score):
    for cut, name in BANDS:
        if score < cut:
            return name
    return "Very low risk"
