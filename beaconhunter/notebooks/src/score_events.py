"""
Scoring CLI for unlabeled events.

Usage:
  python -m src.score_events --input data/beacon_events_eval_unlabeled.csv --output results/eval_scored.csv

Outputs CSV with:
  event_id, host_id, risk_score (0-1), risk_label (HIGH/MED/LOW), top_features (json string)
"""
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from .features import add_derived_features

def load_artifacts():
    sup = joblib.load("artifacts/supervised_model.joblib")
    unsup = joblib.load("artifacts/unsupervised_model.joblib")
    cfg = json.load(open("artifacts/fusion_config.json"))
    return sup, unsup, cfg

def get_feature_names_from_preprocessor(preprocessor):
    """Recreate feature name list (same helper as in training)."""
    try:
        cat_transformer = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = ["protocol", "proc_name_clean", "country_code", "user"]
        cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = []
    num_cols = [
        "bytes_out", "bytes_in", "inter_event_seconds_filled",
        "iev_group_var", "port_rarity_score", "process_risk_score",
        "geo_risk", "dst_port"
    ]
    return cat_names + num_cols

def label_from_score(s):
    if s > 0.8:
        return "HIGH"
    if s > 0.5:
        return "MED"
    return "LOW"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV (unlabeled events)")
    parser.add_argument("--output", required=True, help="Output scored CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df = pd.read_csv(args.input)
    df = add_derived_features(df)

    sup, unsup, cfg = load_artifacts()

    # Prepare features: drop columns not used by model
    X = df.drop(columns=["event_id", "dst_ip", "proc_name"], errors="ignore")

    # Supervised probabilities
    sup_probs = sup.predict_proba(X)[:, 1]

    # Unsupervised normalized anomaly score (use training bounds)
    unsup_raw = -unsup.score_samples(X)
    unsup_min = cfg.get("unsup_min", float(unsup_raw.min()))
    unsup_max = cfg.get("unsup_max", float(unsup_raw.max()))
    unsup_score = (unsup_raw - unsup_min) / (unsup_max - unsup_min + 1e-12)
    unsup_score = np.clip(unsup_score, 0, 1)

    # Combined risk
    s_w = cfg.get("supervised_weight", 0.7)
    u_w = cfg.get("unsupervised_weight", 0.3)
    risk = s_w * sup_probs + u_w * unsup_score
    risk = np.clip(risk, 0.0, 1.0)

    # Extract top contributing features heuristic (top-N feature importances from RF)
    feature_names = []
    top_feature_names = []
    try:
        pre = sup.named_steps["preprocessor"]
        feature_names = get_feature_names_from_preprocessor(pre)
        importances = sup.named_steps["clf"].feature_importances_
        # get top 3 indices
        idx_top = np.argsort(importances)[-3:][::-1]
        top_feature_names = [feature_names[idx] for idx in idx_top if idx < len(feature_names)]
    except Exception:
        top_feature_names = []

    # For each top feature, record its raw (or OHE) value for the row
    top_feature_values = []
    for i in range(len(df)):
        vals = {}
        for fname in top_feature_names:
            # if fname corresponds to numeric column names (end in or equals numeric), map directly
            if fname in df.columns:
                vals[fname] = df.iloc[i][fname]
            else:
                # OHE feature names have format like "proc_name_clean_value"
                parts = fname.split("_", 1)
                col = parts[0]
                # fallback: if original col exists, show its raw
                vals[fname] = df.iloc[i].get(col, None)
        top_feature_values.append(vals)

    # Build output df
    out = pd.DataFrame({
        "event_id": df["event_id"],
        "host_id": df["host_id"],
        "risk_score": risk,
        "risk_label": [label_from_score(s) for s in risk],
        "top_features": [json.dumps(v) for v in top_feature_values]
    })

    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} events -> {args.output}")

if __name__ == "__main__":
    main()
