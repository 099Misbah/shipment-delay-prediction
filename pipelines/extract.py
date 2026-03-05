import pandas as pd
import os

RAW_PATH = "data/raw/incom2024_delay_example_dataset.csv"
BRONZE_PATH = "data/bronze/shipments_bronze.parquet"

def extract_data():
    print("Reading raw dataset...")
    
    df = pd.read_csv(RAW_PATH)
    
    print(f"Raw shape: {df.shape}")
    
    os.makedirs("data/bronze", exist_ok=True)
    
    df.to_parquet(BRONZE_PATH, index=False)
    
    print("Saved to Bronze layer ✔")

if __name__ == "__main__":
    extract_data()