"""
3_basic_analysis.py
Computes core sales KPIs from the cleaned dataset and writes them to outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


OUT_DIR = Path("../outputs")
CLEANED_PATH = OUT_DIR / "cleaned_data.csv"
METRICS_JSON_PATH = OUT_DIR / "basic_kpis.json"
TOP_PRODUCTS_CSV_PATH = OUT_DIR / "top_products_by_sales.csv"


def load_cleaned_dataframe(path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset produced earlier in the pipeline and normalize its columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run preprocess_data.py before computing KPIs.")

    dataframe = pd.read_csv(path, low_memory=False)
    dataframe.columns = [
        column.strip().lower().replace(" ", "_").replace(".", "_")
        for column in dataframe.columns
    ]
    return dataframe


def compute_basic_metrics(dataframe: pd.DataFrame) -> dict[str, object]:
    """
    Calculate core retail KPIs such as total sales and average order value from the dataframe.
    """
    required_sales_column = "sales"
    required_order_column = "order_id"
    required_product_column = "product_name"

    for required_column in (required_sales_column, required_order_column, required_product_column):
        if required_column not in dataframe.columns:
            raise ValueError(f"Column '{required_column}' is required for KPI calculation.")

    total_sales_value = float(dataframe[required_sales_column].sum())
    order_count = dataframe[required_order_column].nunique()
    if order_count == 0:
        raise ValueError("Cannot compute average order value because no orders are present.")
    average_order_value = total_sales_value / order_count

    product_sales = (
        dataframe.groupby(required_product_column)[required_sales_column]
        .sum()
        .sort_values(ascending=False)
    )

    top_products = (
        product_sales.head(10)
        .reset_index()
        .rename(columns={required_product_column: "product_name", required_sales_column: "total_sales"})
    )

    metrics = {
        "total_sales": total_sales_value,
        "average_order_value": average_order_value,
        "order_count": int(order_count),
        "unique_products": int(dataframe[required_product_column].nunique()),
    }

    return {"metrics": metrics, "top_products": top_products}


def write_outputs(metrics_payload: dict[str, object]) -> None:
    """
    Persist computed KPIs and top products to JSON and CSV artefacts for downstream reporting.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = metrics_payload["metrics"]
    top_products = metrics_payload["top_products"]

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    assert isinstance(top_products, pd.DataFrame)
    top_products.to_csv(TOP_PRODUCTS_CSV_PATH, index=False)

    print(f"✅ Saved KPI metrics to {METRICS_JSON_PATH}")
    print(f"✅ Saved top products table to {TOP_PRODUCTS_CSV_PATH}")


def main() -> None:
    """
    Orchestrate the KPI generation workflow for the sales analytics project.
    """
    dataframe = load_cleaned_dataframe(CLEANED_PATH)
    metrics_payload = compute_basic_metrics(dataframe)
    write_outputs(metrics_payload)


if __name__ == "__main__":
    main()

