"""
6_evaluation_and_report.py
Creates summary report (text + some plots saved), and prints recommendations for improvements.
"""
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json

OUT = Path("../outputs")
MR = OUT / "model_results"
MR.mkdir(parents=True, exist_ok=True)

# Load cleaned and clustered data if present
clean = pd.read_csv(OUT / "cleaned_data.csv", low_memory=False)
clean.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in clean.columns]

report = {}
report["n_rows"] = len(clean)
report["columns"] = clean.columns.tolist()

# Classification results summary (if model saved)
cls_models = list(MR.glob("best_classification_model_*.joblib"))
if cls_models:
    model_path = cls_models[0]
    report["classification_model"] = str(model_path)
    try:
        model = joblib.load(model_path)
    except Exception:
        model = None

    # Show feature importances / feature count if pipeline available
    try:
        feat_names = []
        if model is not None and hasattr(model, "named_steps") and "preproc" in model.named_steps:
            preproc = model.named_steps["preproc"]
            # numeric columns (if present)
            try:
                num_cols = preproc.transformers_[0][2]
            except Exception:
                num_cols = []
            # categorical original columns (we won't expand one-hot names here)
            try:
                cat_cols = preproc.transformers_[1][2] if len(preproc.transformers_) > 1 else []
            except Exception:
                cat_cols = []
            feat_names = list(num_cols) + list(cat_cols)
        report["classification_feature_count"] = len(feat_names)
    except Exception:
        # safe fallback
        report["classification_feature_count"] = None

# Clustering summary
clustered_path = OUT / "clustered_data.csv"
if clustered_path.exists():
    clustered = pd.read_csv(clustered_path, low_memory=False)
    # accept either 'cluster' or 'cluster_label'
    cluster_col = None
    for possible in ("cluster", "cluster_label"):
        if possible in clustered.columns:
            cluster_col = possible
            break
    if cluster_col:
        counts = clustered[cluster_col].value_counts().to_dict()
        report["cluster_counts"] = counts
        # Save a quick clusters scatter (sales vs profit)
        if "sales" in clustered.columns and "profit" in clustered.columns:
            try:
                plt.figure(figsize=(8,6))
                sns.scatterplot(data=clustered, x="sales", y="profit", hue=cluster_col, palette="tab10", alpha=0.6)
                plt.title("Clusters: Sales vs Profit")
                plt.tight_layout()
                plt.savefig(MR / "clusters_sales_profit.png")
                plt.clf()
            except Exception:
                # if plotting fails for any reason, continue without crashing
                pass

# Save report json and a human-readable summary
# Use UTF-8 and ensure_ascii=False so unicode characters are preserved
with open(MR / "summary_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Minimal plain text report — write with utf-8 encoding to avoid UnicodeEncodeError on Windows
with open(MR / "summary_report.txt", "w", encoding="utf-8") as f:
    f.write("Project summary\n")
    f.write("=================\n")
    f.write(f"Rows in cleaned data: {report.get('n_rows')}\n")
    f.write("Columns:\n")
    for c in report.get("columns", []):
        f.write(f" - {c}\n")
    f.write("\nCluster counts (if available):\n")
    for k, v in report.get("cluster_counts", {}).items():
        f.write(f"Cluster {k}: {v} rows\n")
    f.write("\nClassification model: " + str(report.get("classification_model", "None")) + "\n")

print("✅ Summary reports saved to", MR)
