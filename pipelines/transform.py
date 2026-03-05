import pandas as pd

BRONZE_PATH = "data/bronze/shipments_bronze.parquet"
SILVER_PATH = "data/silver/shipments_silver.parquet"

def main():
    df = pd.read_parquet(BRONZE_PATH)
    print("Loaded bronze:", df.shape)

    # Standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Convert shipping_date if exists
    if "shipping_date" in df.columns:
        df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce")

    # Remove duplicates
    df = df.drop_duplicates()

    # Example: fix negative or weird values if columns exist
    for col in ["product_price", "profit_per_order"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(SILVER_PATH, index=False)
    print("✅ Saved silver:", df.shape)

if __name__ == "__main__":
    main()