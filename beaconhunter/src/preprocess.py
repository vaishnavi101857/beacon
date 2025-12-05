from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor():
    """
    Returns a ColumnTransformer pipeline that imputes and encodes numeric and categorical features.
    The caller must ensure the dataframe contains the derived features created in features.add_derived_features.
    """
    categorical_cols = ["protocol", "proc_name_clean", "country_code", "user"]
    numeric_cols = [
        "bytes_out", "bytes_in", "inter_event_seconds_filled",
        "iev_group_var", "port_rarity_score", "process_risk_score",
        "geo_risk", "dst_port"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ], remainder="drop")

    return preprocessor
