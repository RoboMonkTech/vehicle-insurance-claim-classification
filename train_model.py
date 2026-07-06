# ==========================================================
# VEHICLE INSURANCE CLAIM CLASSIFICATION SYSTEM
# FINAL TRAINING PIPELINE
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset/vehicle_insurance.csv")

df.drop("id", axis=1, inplace=True)

X = df.drop("Response", axis=1)
y = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# FEATURES
# ==========================================================

categorical_features = ["Gender", "Vehicle_Age", "Vehicle_Damage"]

numerical_features = [
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage"
]

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# ==========================================================
# MODELS
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = []

best_model = None
best_pipeline = None
best_f1 = 0

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)

    results.append([name, acc, prec, rec, f1, roc])

    print(f"\n{name}")
    print("-" * 50)
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC AUC  :", round(roc, 4))

    if f1 > best_f1:
        best_f1 = f1
        best_pipeline = pipeline
        best_model = name

# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
)

print("\n" + "=" * 80)
print("MODEL RANKING")
print("=" * 80)
print(results_df.sort_values(by="F1", ascending=False))

print("\nBEST MODEL:", best_model)
print("BEST F1:", best_f1)

# ==========================================================
# THRESHOLD TUNING
# ==========================================================

print("\n" + "=" * 80)
print("THRESHOLD TUNING")
print("=" * 80)

probs = best_pipeline.predict_proba(X_test)[:, 1]

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

best_threshold = 0.5
best_thr_f1 = 0

for t in thresholds:

    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)

    print(f"Threshold {t} -> F1: {round(f1, 4)}")

    if f1 > best_thr_f1:
        best_thr_f1 = f1
        best_threshold = t

print("\nBEST THRESHOLD:", best_threshold)
print("BEST THRESHOLD F1:", best_thr_f1)

# ==========================================================
# FINAL EVALUATION
# ==========================================================

final_preds = (probs >= best_threshold).astype(int)

print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

print(classification_report(y_test, final_preds))

print("Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))

print("\nAccuracy :", accuracy_score(y_test, final_preds))
print("Precision:", precision_score(y_test, final_preds, zero_division=0))
print("Recall   :", recall_score(y_test, final_preds, zero_division=0))
print("F1 Score :", f1_score(y_test, final_preds, zero_division=0))
print("ROC AUC  :", roc_auc_score(y_test, probs))

# ==========================================================
# SAVE MODEL + THRESHOLD
# ==========================================================

joblib.dump(
    {
        "model": best_pipeline,
        "threshold": best_threshold
    },
    "models/insurance_pipeline.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY")
print("LOCATION: models/insurance_pipeline.pkl")