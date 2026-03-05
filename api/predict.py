import pandas as pd
import joblib

MODEL_PATH = "modeling/model.pkl"

model = joblib.load(MODEL_PATH)

def _get_feature_cols():
    """
    Pull the exact columns the trained preprocessor expects.
    Works because your model is a Pipeline: ("prep", ColumnTransformer) -> ("model", estimator)
    """
    prep = model.named_steps["prep"]
    # transformers_: [("cat", OneHotEncoder, [cols...]), ("num", "passthrough", [cols...])]
    cat_cols = prep.transformers_[0][2]
    num_cols = prep.transformers_[1][2]
    return list(cat_cols), list(num_cols)

CAT_COLS, NUM_COLS = _get_feature_cols()

def _coerce_like_training(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure all expected columns exist
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "missing"
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = 0

    # --- Categorical: string + fill missing ---
    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("missing")

    # --- Numeric: handle numbers + datetime strings ---
    for c in NUM_COLS:
        # if it comes as a string like "2015-08-13 00:00:00+01:00"
        if df[c].dtype == "object" or str(df[c].dtype).startswith("string"):
            # try parse datetime first
            dt = pd.to_datetime(df[c], errors="coerce", utc=True)
            if dt.notna().any():
                # convert to int64 timestamp (nanoseconds)
                df[c] = dt.view("int64")
            else:
                # otherwise numeric coercion
                df[c] = pd.to_numeric(df[c], errors="coerce")

        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df[c] = df[c].fillna(0)

    # keep only expected columns (extra keys in payload won’t break)
    df = df[CAT_COLS + NUM_COLS]
    return df

def predict(payload: dict):
    X = pd.DataFrame([payload])
    X = _coerce_like_training(X)

    pred = int(model.predict(X)[0])

    # if binary classifier
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0, 1])

    return {"prediction": pred, "probability_delay": proba}