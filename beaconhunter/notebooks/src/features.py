import pandas as pd
import numpy as np

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived security features to the dataframe. Returns a new dataframe.
    Requires these input columns to exist: host_id, dst_ip, dst_port, inter_event_seconds, proc_name, country_code.
    """
    df = df.copy()

    # fill inter_event_seconds with median for later calculations
    df["inter_event_seconds_filled"] = df["inter_event_seconds"].fillna(df["inter_event_seconds"].median())

    # 1) Inter-event variability per (host_id, dst_ip) - beacon traffic often very regular (low variance)
    df["iev_group_var"] = df.groupby(["host_id", "dst_ip"])["inter_event_seconds_filled"].transform("std")
    # If a group has a single event, std will be NaN -> fill with global median
    df["iev_group_var"] = df["iev_group_var"].fillna(df["iev_group_var"].median())

    # 2) Port rarity score: rare ports get higher score (1 = rarest)
    port_counts = df["dst_port"].value_counts()
    # Normalize counts to [0,1], then rarity = 1 - normalized_freq
    norm_freq = (port_counts / port_counts.max()).to_dict()
    df["port_rarity_score"] = df["dst_port"].map(lambda p: 1.0 - norm_freq.get(p, 0.0))

    # 3) Process risk score: map suspicious processes to higher risk
    risky_processes = {
        "powershell.exe": 1.0,
        "rundll32.exe": 0.95,
        "sliver-client.exe": 1.0,
        "cmd.exe": 0.75,
        "wmic.exe": 0.7,
        "cscript.exe": 0.7
    }
    df["proc_name_clean"] = df["proc_name"].fillna("").str.lower().str.strip()
    df["process_risk_score"] = df["proc_name_clean"].map(lambda p: risky_processes.get(p, 0.1))

    # 4) GeoIP risk: map country to risk buckets (higher = more suspicious)
    geo_risk = {
        "RU": 1.0, "CN": 1.0, "IR": 0.95, "KP": 1.0,
        "UA": 0.7, "DE": 0.6, "BR": 0.6,
        "US": 0.1, "GB": 0.1, "CA": 0.1
    }
    df["country_code"] = df["country_code"].fillna("ZZ")
    df["geo_risk"] = df["country_code"].map(lambda c: geo_risk.get(c, 0.5))

    # Ensure numeric columns exist and fill NAs
    for col in ["bytes_out", "bytes_in", "dst_port"]:
        if col not in df.columns:
            df[col] = 0
    df["bytes_out"] = df["bytes_out"].fillna(0)
    df["bytes_in"] = df["bytes_in"].fillna(0)

    # Drop intermediate helper if not desired (keep proc_name_clean)
    return df
