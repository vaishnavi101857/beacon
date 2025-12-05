"""
Train pipeline for BeaconHunter.
Usage:
    python -m src.train_detector
Produces artifacts/:
  - artifacts/supervised_model.joblib
  - artifacts/unsupervised_model.joblib
  - artifacts/fusion_config.json
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report
from sklearn.pipeline import Pipeline

from .features import add_derived_features
from .preprocess import build_preprocessor

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def get_feature_names_from_preprocessor(preprocessor):
    """
    Return list of feature names produced by the preprocessor (order matches transformed columns).
    Works for OneHotEncoder and numeric columns defined in preprocess.build_preprocessor.
    """
    cat_transformer = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    # categorical base names used in preprocess.py
    cat_cols = ["protocol", "proc_name_clean", "country_code", "user"]
    try:
        cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
    except Exception:
        # fallback older sklearn
        cat_names = []
    num_cols = [
        "bytes_out", "bytes_in", "inter_event_seconds_filled",
        "iev_group_var", "port_rarity_score", "process_risk_score",
        "geo_risk", "dst_port"
    ]
    return cat_names + num_cols

def main():
    # 1. Load train data and add derived features
    df = pd.read_csv("data/beacon_events_train.csv")
    df = add_derived_features(df)
    # Keep only expected columns + label
    if "label" not in df.columns:
        raise RuntimeError("Training CSV must include 'label' column.")

    X = df.drop(columns=["label", "event_id", "dst_ip", "proc_name"])  # drop some non-modeled columns; keep proc_name_clean
    y = df["label"].astype(int)

    # 2. Train / validation split stratified by label
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    # 3. Build preprocessor
    preprocessor = build_preprocessor()

    # 4. Supervised model pipeline
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    sup_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", clf)])
    sup_pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    val_probs = sup_pipeline.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    print(f"Supervised ROC-AUC: {roc_auc:.4f}")
    print(f"Supervised PR-AUC: {pr_auc:.4f}")

    # Choose operating threshold by maximizing F1 on validation set
    prec, rec, thresh = precision_recall_curve(y_val, val_probs)
    # compute F1 for each threshold (length thresh = len(prec)-1)
    f1_scores = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    if best_idx >= len(thresh):
        operating_threshold = 0.5
    else:
        operating_threshold = float(thresh[best_idx])
    print(f"Chosen operating threshold (validation max F1): {operating_threshold:.3f} (F1={f1_scores[best_idx]:.3f})")
    # print classification report at chosen threshold
    val_pred_label = (val_probs > operating_threshold).astype(int)
    print(classification_report(y_val, val_pred_label))

    # 5. Train unsupervised model (IsolationForest) on benign events only
    benign_df = df[df["label"] == 0].drop(columns=["label", "event_id", "dst_ip", "proc_name"], errors="ignore")
    unsup_pre = build_preprocessor()
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    unsup_pipeline = Pipeline(steps=[("preprocessor", unsup_pre), ("iso", iso)])
    # Fit on benign data
    unsup_pipeline.fit(benign_df)

    # compute anomaly raw scores on benign training set for consistent normalization
    # (we use -score_samples so higher => more anomalous)
    # Note: score_samples accepts the original (non-transformed) features because pipeline contains preprocessor
    unsup_raw_train = -unsup_pipeline.score_samples(benign_df)
    unsup_min = float(unsup_raw_train.min())
    unsup_max = float(unsup_raw_train.max())
    print(f"Unsup normalization bounds: min={unsup_min:.4f}, max={unsup_max:.4f}")

    # 6. Save artifacts
    joblib.dump(sup_pipeline, f"{ARTIFACT_DIR}/supervised_model.joblib")
    joblib.dump(unsup_pipeline, f"{ARTIFACT_DIR}/unsupervised_model.joblib")

    # Save metadata (weights, thresholds, normalization bounds, top features)
    try:
        feature_names = get_feature_names_from_preprocessor(preprocessor)
        importances = sup_pipeline.named_steps["clf"].feature_importances_
        # pick top 10 features for reference
        top_idx = list(importances.argsort()[-10:][::-1])
        top_features = [feature_names[i] for i in top_idx if i < len(feature_names)]
    except Exception:
        top_features = []

    fusion_config = {
        "supervised_weight": 0.7,
        "unsupervised_weight": 0.3,
        "operating_threshold": operating_threshold,
        "unsup_min": unsup_min,
        "unsup_max": unsup_max,
        "top_features": top_features
    }
    with open(f"{ARTIFACT_DIR}/fusion_config.json", "w") as fh:
        json.dump(fusion_config, fh, indent=2)

    print("Artifacts saved in", ARTIFACT_DIR)
    print("Top features (reference):", fusion_config["top_features"])

if __name__ == "__main__":
    main()
