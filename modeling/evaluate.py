import pandas as pd
import joblib

MODEL_PATH = "modeling/model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess_input(payload: dict) -> pd.DataFrame:
    X = pd.DataFrame([payload]).copy()

    # datetime -> int64 if any datetime strings come in
    for col in X.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                X[col] = pd.to_datetime(X[col], errors="ignore")
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    X[col] = X[col].view("int64")
            except Exception:
                pass

    # Fill missing
    for c in X.columns:
        if X[c].dtype == "O":
            X[c] = X[c].fillna("missing")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X

def predict(payload: dict):
    X = preprocess_input(payload)
    pred = int(model.predict(X)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])

    return {"prediction": pred, "delay_risk_score": proba}