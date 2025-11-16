"""
quick_inspect.py
Provides a lightweight dataset inspection using the repository data path.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "superstore.csv"


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the raw Superstore dataset for a quick diagnostic check.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset at {path}.")
    dataframe = pd.read_csv(path, low_memory=False)
    dataframe.columns = [
        column.strip().lower().replace(" ", "_").replace(".", "_")
        for column in dataframe.columns
    ]
    return dataframe


def main() -> None:
    """
    Display key structural information about the dataset for manual review.
    """
    dataframe = load_dataset(DATA_PATH)
    print("Shape:", dataframe.shape)
    print("Columns:", dataframe.columns.tolist())
    print("Preview:\n", dataframe.head(10).T)


if __name__ == "__main__":
    main()
