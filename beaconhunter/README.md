BeaconHunter - C2 Beacon Detection (Student-style pipeline)

Prereqs:
- Put your CSV files in data/: beacon_events_train.csv, beacon_events_test_labeled.csv, beacon_events_eval_unlabeled.csv

Install:
pip install -r requirements.txt

Run training (creates artifacts/):
python -m src.train_detector

Evaluate on labeled test set:
python -m src.evaluate_detector

Score unlabeled events:
python -m src.score_events --input data/beacon_events_eval_unlabeled.csv --output results/eval_scored.csv

Notes:
- artifacts/ will contain supervised_model.joblib, unsupervised_model.joblib, fusion_config.json
- results/eval_scored.csv contains event_id, host_id, risk_score, risk_label, top_features
