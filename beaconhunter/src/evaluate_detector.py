"""
Evaluate saved detector on labeled test data.
Usage:
  python -m src.evaluate_detector
Prints: ROC-AUC, PR-AUC, confusion matrix, classification report, and top-10 misclassified events.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report

from .features import add_derived_features

def main():
    # Load test data and add derived features
    df = pd.read_csv("data/beacon_events_test_labeled.csv")
    df = add_derived_features(df)

    # Load artifacts
    sup = joblib.load("artifacts/supervised_model.joblib")
    unsup = joblib.load("artifacts/unsupervised_model.joblib")
    cfg = json.load(open("artifacts/fusion_config.json"))

    # Prepare features (drop label only for model input)
    X = df.drop(columns=["label", "event_id", "dst_ip", "proc_name"], errors="ignore")
    y = df["label"].astype(int)

    # Supervised probabilities
    sup_probs = sup.predict_proba(X)[:, 1]

    # Unsupervised anomaly score (use stored normalization bounds)
    unsup_raw = -unsup.score_samples(X)
    unsup_min = cfg.get("unsup_min", float(unsup_raw.min()))
    unsup_max = cfg.get("unsup_max", float(unsup_raw.max()))
    unsup_score = (unsup_raw - unsup_min) / (unsup_max - unsup_min + 1e-12)
    unsup_score = np.clip(unsup_score, 0, 1)

    # Combined risk
    s_w = cfg.get("supervised_weight", 0.7)
    u_w = cfg.get("unsupervised_weight", 0.3)
    combined = s_w * sup_probs + u_w * unsup_score

    # Metrics
    roc_auc = roc_auc_score(y, combined)
    pr_auc = average_precision_score(y, combined)
    threshold = cfg.get("operating_threshold", 0.5)
    preds = (combined > threshold).astype(int)

    print(f"Combined ROC-AUC: {roc_auc:.4f}")
    print(f"Combined PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix (at threshold {:.3f}):".format(threshold))
    print(confusion_matrix(y, preds))
    print("Classification report (at threshold {:.3f}):".format(threshold))
    print(classification_report(y, preds))

    # Top 10 misclassified: high-risk benign and low-risk malicious
    result_df = df.copy()
    result_df["risk_score"] = combined
    result_df["pred"] = preds
    misclassified = result_df[result_df["pred"] != result_df["label"]].copy()
    # sort by risk for human inspection (high risk but benign first; low risk but malicious next)
    misclassified = misclassified.sort_values(by="risk_score", ascending=False).head(10)
    display_cols = ["event_id", "host_id", "dst_ip", "dst_port", "proc_name", "bytes_out",
                    "inter_event_seconds_filled", "iev_group_var", "process_risk_score",
                    "geo_risk", "risk_score", "label"]
    print("\nTop misclassified events (for investigation):")
    print(misclassified[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
