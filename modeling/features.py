import pandas as pd

SILVER_PATH = "data/silver/shipments_silver.parquet"
GOLD_PATH = "data/gold/shipments_gold.parquet"

def main():
    df = pd.read_parquet(SILVER_PATH)
    print("Loaded silver:", df.shape)

    # Example features
    if "shipping_date" in df.columns:
        df["ship_year"] = df["shipping_date"].dt.year
        df["ship_month"] = df["shipping_date"].dt.month
        df["ship_dayofweek"] = df["shipping_date"].dt.dayofweek

    df.to_parquet(GOLD_PATH, index=False)
    print("✅ Saved gold:", df.shape)

if __name__ == "__main__":
    main()