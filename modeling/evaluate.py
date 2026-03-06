import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

GOLD_PATH = "data/gold/shipments_gold.parquet"
MODEL_PATH = "modeling/model.pkl"
FEATURES_PATH = "modeling/feature_columns.json"


def fix_types_like_train(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # 1) Convert datetime columns to numeric (int64 timestamp)
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        X[c] = X[c].view("int64")

    # 2) Categorical columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 3) Everything else numeric
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 4) Fill missing
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("missing").astype(str)
    if num_cols:
        X[num_cols] = X[num_cols].fillna(0)

    return X


def main():
    df = pd.read_parquet(GOLD_PATH)

    if "label" not in df.columns:
        raise ValueError("label column not found in gold dataset!")

    y = df["label"]
    X = df.drop(columns=["label"])

    # Align columns to what training used (important!)
    try:
        with open(FEATURES_PATH, "r") as f:
            feature_cols = json.load(f)

        for col in feature_cols:
            if col not in X.columns:
                X[col] = None  # add missing columns

        X = X[feature_cols]  # exact same order as training
    except FileNotFoundError:
        print("⚠️ feature_columns.json not found — continuing without column alignment.")

    # ✅ Apply same type fix as training
    X = fix_types_like_train(X)

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)

    print("\n✅ Classification Report:\n")
    print(classification_report(y, preds))

    if set(y.unique()) <= {0, 1} and hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
        print("✅ ROC-AUC:", roc_auc_score(y, prob))


if __name__ == "__main__":
    main()