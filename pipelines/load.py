import pandas as pd
import sqlite3

SILVER_PATH = "data/silver/shipments_silver.parquet"
DB_PATH = "data/shipments.db"
TABLE_NAME = "shipments_silver"

def main():
    df = pd.read_parquet(SILVER_PATH)
    print("Loaded silver:", df.shape)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print(f"✅ Loaded into SQLite: {DB_PATH} table={TABLE_NAME}")

if __name__ == "__main__":
    main()