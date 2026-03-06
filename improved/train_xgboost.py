import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

GOLD_PATH = "data/gold/shipments_gold.parquet"
MODEL_PATH = "improved/model_xgboost.pkl"
FEATURES_PATH = "improved/feature_columns_xgboost.json"
BEST_PARAMS_PATH = "improved/best_params_xgboost.json"


def fix_types_like_train(X: pd.DataFrame):
    X = X.copy()

    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        X[c] = X[c].view("int64")

    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("missing").astype(str)
    if num_cols:
        X[num_cols] = X[num_cols].fillna(0)

    return X, cat_cols, num_cols


def main():
    df = pd.read_parquet(GOLD_PATH)

    if "label" not in df.columns:
        raise ValueError("label column not found!")

    label_map = {-1: 0, 0: 1, 1: 2}
    y = df["label"].map(label_map)
    X = df.drop(columns=["label"]).copy()

    X, cat_cols, num_cols = fix_types_like_train(X)

    with open(FEATURES_PATH, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(sorted(y.unique())),
        eval_metric="mlogloss",
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=10,
        scoring="f1_weighted",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train_res, y_train_res)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test_enc)

    print("\n✅ Best Parameters:\n", search.best_params_)
    print("\n✅ Accuracy:", accuracy_score(y_test, preds))
    print("✅ Weighted F1:", f1_score(y_test, preds, average="weighted"))
    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, preds))

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(search.best_params_, f, indent=2)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": best_model,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
        },
        MODEL_PATH,
    )
    print(f"\n✅ Saved improved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()