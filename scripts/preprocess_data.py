"""
2_preprocess_data.py
Full preprocessing (the five cleaning steps) and feature engineering.
Saves cleaned dataset to outputs/cleaned_data.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("../outputs/step1_loaded.csv")
OUT_PATH = Path("../outputs")
OUT_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_PATH, low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

n_before = len(df)
df = df.drop_duplicates()
print(f"Duplicates removed: {n_before - len(df)}")

possible_date_cols = [c for c in df.columns if "date" in c or "order" in c and "date" in c]
for c in df.columns:
    if "date" in c or c in ("order_date","orderdate","ship_date","shipdate"):
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            print(f"Parsed datetime: {c} (nulls after parse: {df[c].isnull().sum()})")
        except Exception:
            pass


for c in df.select_dtypes(include=['object']).columns:
    df[c] = df[c].astype(str).str.strip()


num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if "quantity" in df.columns:
    neg_q = (df["quantity"] <= 0).sum()
    if neg_q > 0:
        print("Found non-positive quantity rows:", neg_q)
        df = df[df["quantity"] > 0]

if "sales" in df.columns:
    df = df[pd.to_numeric(df["sales"], errors='coerce').notnull()]
    df["sales"] = df["sales"].astype(float)
if "profit" in df.columns:
    df["profit"] = pd.to_numeric(df["profit"], errors='coerce').fillna(0.0)


for c in df.select_dtypes(include=[np.number]).columns:
    nmiss = df[c].isnull().sum()
    if nmiss:
        med = df[c].median()
        df[c].fillna(med, inplace=True)
        print(f"Imputed numeric {c} with median {med} (filled {nmiss})")


for c in df.select_dtypes(include=['object']).columns:
    nmiss = df[c].isnull().sum()
    if nmiss:
        df[c].fillna("Unknown", inplace=True)
        print(f"Imputed categorical {c} with 'Unknown' (filled {nmiss})")

if "order_date" in df.columns:
    n_missing_dates = df["order_date"].isnull().sum()
    if n_missing_dates > 0:
        print(f"Order date missing in {n_missing_dates} rows — dropping them.")
        df = df[df["order_date"].notnull()]


if ("profit" in df.columns) and ("sales" in df.columns):
    df["profit_margin"] = df.apply(lambda r: (r["profit"]/r["sales"]) if r["sales"] and r["sales"] != 0 else 0.0, axis=1)
    df["is_profitable"] = (df["profit"] > 0).astype(int)

if "order_date" in df.columns:
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["order_week"] = df["order_date"].dt.isocalendar().week


for c in ["category","sub_category","sub.category","sub_category"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.title()


clean_path = OUT_PATH / "cleaned_data.csv"
df.to_csv(clean_path, index=False)
print(f"✅ Cleaned dataset saved to {clean_path}")
print("Final shape:", df.shape)
