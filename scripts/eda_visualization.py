"""
3_eda_visualization_fixed.py
Performs EDA and saves charts to outputs/eda_charts/
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib


OUT = Path("../outputs")
EDACHARTS = OUT / "eda_charts"
EDACHARTS.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(OUT / "cleaned_data.csv", low_memory=False)


df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]


if "记录数" in df.columns:
    df = df.rename(columns={"记录数": "record_count"})


matplotlib.rcParams['font.family'] = 'Arial'  


if "category" in df.columns and "sales" in df.columns:
    cat_summary = df.groupby("category").agg(
        total_sales=("sales","sum"), total_profit=("profit","sum")
    ).sort_values("total_sales", ascending=False)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=cat_summary.index, y=cat_summary["total_sales"])
    plt.title("Total Sales by Category")
    plt.ylabel("Sales")
    plt.xlabel("Category")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(EDACHARTS / "sales_by_category.png")
    plt.clf()


if "order_date" in df.columns and "sales" in df.columns:
    df["order_date"] = pd.to_datetime(df["order_date"])
    monthly = df.set_index("order_date").resample("M").agg({"sales":"sum","profit":"sum"})
    
    plt.figure(figsize=(10,4))
    plt.plot(monthly.index, monthly["sales"], marker='o')
    plt.title("Monthly Sales Trend")
    plt.tight_layout()
    plt.savefig(EDACHARTS / "monthly_sales_trend.png")
    plt.clf()

numeric_cols = [c for c in ["sales","profit","profit_margin","record_count"] if c in df.columns]
for c in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[c].dropna(), bins=50, kde=True)
    plt.title(f"Distribution of {c}")
    plt.tight_layout()
    plt.savefig(EDACHARTS / f"dist_{c}.png")
    plt.clf()


num_df = df.select_dtypes(include=["number"])
if num_df.shape[1] >= 2:
    plt.figure(figsize=(8,6))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(EDACHARTS / "correlation_heatmap.png")
    plt.clf()

print("✅ EDA charts saved to", EDACHARTS)
