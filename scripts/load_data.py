"""
1_load_and_inspect.py
Loads the CSV and prints a concise data summary. Saves a copy used for further steps.
"""
import pandas as pd
from pathlib import Path
import json

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "superstore.csv"
OUT_PATH = Path("../outputs")
OUT_PATH.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"{DATA_PATH} not found. Put your CSV at data/superstore.csv or run 0_kaggle_download.py")

# Load
df = pd.read_csv(DATA_PATH, low_memory=False)

# Normalize column names (lowercase, underscores)
df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

# Basic inspection
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nHead (first 5 rows):")
print(df.head().T)

# Show dtypes and null counts
info = pd.DataFrame({"dtype": df.dtypes.astype(str), "null_count": df.isnull().sum()})
print("\nColumn info (dtype + nulls):")
print(info)

# Save a snapshot for next step
df.to_csv(OUT_PATH / "step1_loaded.csv", index=False)
print("\nSaved snapshot to outputs/step1_loaded.csv")
# Also save column info for quick reference
(info.reset_index().rename(columns={"index":"column"}).to_json(OUT_PATH / "column_info.json", orient="records"))
