import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

GOLD_PATH = "data/gold/shipments_gold.parquet"
MODEL_PATH = "modeling/model.pkl"


def main():
    df = pd.read_parquet(GOLD_PATH)

    # Target
    if "label" not in df.columns:
        raise ValueError("❌ label column not found!")

    y = df["label"]
    X = df.drop(columns=["label"]).copy()

    
    import json, os

    FEATURES_PATH = "modeling/feature_columns.json"
    os.makedirs("modeling", exist_ok=True)

    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=2)
    print(f"✅ Saved feature columns -> {FEATURES_PATH}")
    
    # --- FIX TYPES (prevents float + object stacking error) ---
    # 1) Convert datetime columns to numeric (timestamp)
    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        # Convert datetime to int64 nanoseconds
        X[c] = X[c].view("int64")

    # 2) Define categorical columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 3) Everything else is numeric; force numeric dtype
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 4) Fill missing values
    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("missing")
    if num_cols:
        X[num_cols] = X[num_cols].fillna(0)
    # --- END FIX TYPES ---
    import json
    from pathlib import Path

    FEATURES_PATH = Path("modeling/feature_columns.json")

    # after X is finalized (types fixed + missing filled)
    FEATURES_PATH.write_text(json.dumps(list(X.columns), indent=2))
    print(f"✅ Saved feature columns -> {FEATURES_PATH}")
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=500)

    clf = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, preds))

    # ROC-AUC (binary only)
    if set(y.unique()) <= {0, 1}:
        prob = clf.predict_proba(X_test)[:, 1]
        print("✅ ROC-AUC:", roc_auc_score(y_test, prob))

    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()