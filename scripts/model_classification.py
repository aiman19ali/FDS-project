"""
4_model_classification.py
Builds classification models to predict binary target 'is_profitable' (created in preprocess step).
Includes baseline logistic regression and RandomForest, with cross-validation and a simple grid search.
Saves best pipeline and evaluation metrics to outputs/model_results/
"""
import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

# Paths
OUT = Path("../outputs")
MR = OUT / "model_results"
MR.mkdir(parents=True, exist_ok=True)

# Load cleaned data
df = pd.read_csv(OUT / "cleaned_data.csv", low_memory=False)
df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]

# Ensure target exists
if "is_profitable" not in df.columns:
    df["is_profitable"] = (df["profit"] > 0).astype(int)

# Select features automatically: numeric + a few categorical (if available)
num_features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("is_profitable","row_id")]
cat_features = [c for c in ["category","sub_category","segment","ship_mode","market","state"] if c in df.columns]

# Keep a reasonable small set for faster runs
num_features = [f for f in num_features if f in ("sales","profit","profit_margin","quantity","discount","shipping_cost","order_month","order_dayofweek")]
num_features = [f for f in num_features if f in df.columns]
cat_features = [f for f in cat_features if f in df.columns]

print("Numeric features:", num_features)
print("Categorical features:", cat_features)

X = df[num_features + cat_features].copy()
y = df["is_profitable"]

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessing pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # ✅ fixed
])

preproc = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features)
], remainder="drop")

# Models to compare
models = {
    "logistic": Pipeline([
        ("preproc", preproc),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "rf": Pipeline([
        ("preproc", preproc),
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ])
}

results = {}
for name, pipe in models.items():
    print(f"\nTraining {name} ...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
    print(f"{name} CV F1 scores:", scores, "mean:", scores.mean())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps["clf"], "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    results[name] = {"model": pipe, "acc": acc, "f1_cv_mean": scores.mean(), "report": report, "roc_auc": roc}
    print(f"\n{name} test accuracy: {acc}, roc_auc: {roc}")
    print(report)

# Save best model by F1 CV mean
best_name = max(results.keys(), key=lambda k: results[k]["f1_cv_mean"])
best_model = results[best_name]["model"]
joblib.dump(best_model, MR / f"best_classification_model_{best_name}.joblib")
print(f"Saved best classification model ({best_name}) to {MR}")

# Simple grid search to optimize RF (if available)
if "rf" in models:
    print("Running small GridSearch for RandomForest...")
    rf_pipe = models["rf"]
    params = {"clf__n_estimators": [100, 200], "clf__max_depth":[None, 10, 20]}
    gs = GridSearchCV(rf_pipe, params, cv=3, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    print("GridSearch best params:", gs.best_params_)
    joblib.dump(gs.best_estimator_, MR / "rf_gridsearch_best.joblib")
    print("Saved rf_gridsearch_best.joblib")
