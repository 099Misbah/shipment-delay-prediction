import pandas as pd

BRONZE_PATH = "data/bronze/shipments_bronze.parquet"

def main():
    df = pd.read_parquet(BRONZE_PATH)
    print("✅ Loaded bronze:", df.shape)

    # Basic checks
    assert df.shape[0] > 0, "Dataset is empty"
    assert df.shape[1] > 0, "No columns found"

    # Duplicate check
    dup = df.duplicated().sum()
    print("Duplicates:", dup)

    # Missing values check (top 10)
    missing = df.isna().sum().sort_values(ascending=False).head(10)
    print("\nTop missing columns:\n", missing)

    # Label check
    if "label" in df.columns:
        print("\nLabel distribution:\n", df["label"].value_counts(dropna=False))

    print("\n✅ Validation complete.")

if __name__ == "__main__":
    main()