import pandas as pd

SILVER_PATH = "data/silver/shipments_silver.parquet"
GOLD_PATH = "data/gold/shipments_gold.parquet"

def main():
    df = pd.read_parquet(SILVER_PATH)
    print("Loaded silver:", df.shape)

    # standardize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # shipping_date features
    if "shipping_date" in df.columns:
        df["shipping_date"] = pd.to_datetime(df["shipping_date"], errors="coerce", utc=True)

        if pd.api.types.is_datetime64_any_dtype(df["shipping_date"]):
            df["ship_year"] = df["shipping_date"].dt.year
            df["ship_month"] = df["shipping_date"].dt.month
            df["ship_dayofweek"] = df["shipping_date"].dt.dayofweek

    # order_date features
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", utc=True)

        if pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_year"] = df["order_date"].dt.year
            df["order_month"] = df["order_date"].dt.month
            df["order_dayofweek"] = df["order_date"].dt.dayofweek

    # business features
    if "sales" in df.columns and "order_item_quantity" in df.columns:
        df["sales_per_item"] = df["sales"] / df["order_item_quantity"].replace(0, 1)

    if "profit_per_order" in df.columns and "sales_per_customer" in df.columns:
        df["profit_ratio_est"] = df["profit_per_order"] / df["sales_per_customer"].replace(0, 1)

    df.to_parquet(GOLD_PATH, index=False)
    print("✅ Saved gold:", df.shape)

if __name__ == "__main__":
    main()