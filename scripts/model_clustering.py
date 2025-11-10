

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

OUT = Path("../outputs")
MR = OUT / "model_results"
MR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "cleaned_data.csv", low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

# Choose numeric features for clustering
features = []
for f in ["sales","profit","profit_margin","quantity","discount","shipping_cost"]:
    if f in df.columns:
        features.append(f)
if len(features) < 2:
    raise RuntimeError("Not enough numeric features for clustering: found " + str(features))

X = df[features].fillna(0).values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Determine K using silhouette for k=2..8
best_k = 2
best_score = -1
scores = {}
for k in range(2,9):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    db = davies_bouldin_score(Xs, labels)
    scores[k] = {"silhouette": sil, "db": db}
    print(f"k={k}: silhouette={sil:.4f}, db={db:.4f}")
    if sil > best_score:
        best_score = sil
        best_k = k

print(f"Best k by silhouette: {best_k} (score {best_score:.4f})")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)
labels = kmeans.labels_

# Save clustered dataframe
df["cluster"] = labels
df.to_csv(OUT / "clustered_data.csv", index=False)
joblib.dump({"scaler": scaler, "kmeans": kmeans, "features": features}, MR / "kmeans_artifact.joblib")
print("Saved clustered_data.csv and kmeans_artifact.joblib")