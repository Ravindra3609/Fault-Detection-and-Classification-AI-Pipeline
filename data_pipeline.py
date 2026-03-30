"""
data_pipeline.py
================
NASA CMAPSS turbofan degradation dataset pipeline.
- Download / cache dataset
- Feature extraction from raw sensor traces
- Train/test split with RUL labeling
- Fault labeling (last N cycles = fault zone)
- Normalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import urllib.request
import zipfile
import io

DATA_DIR = Path("data")

# ── CMAPSS column names ───────────────────────────────────────────────────────
INDEX_COLS = ["unit_id", "cycle"]
OP_COLS    = ["op_1", "op_2", "op_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]   # 21 sensors
ALL_COLS   = INDEX_COLS + OP_COLS + SENSOR_COLS

# Sensors with near-zero variance in FD001 (constant readings — drop them)
USELESS_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
USEFUL_SENSORS  = [s for s in SENSOR_COLS if s not in USELESS_SENSORS]

# Fault zone threshold: last FAULT_CYCLES cycles of each unit = fault state
FAULT_CYCLES = 30


def download_cmapss(data_dir: Path = DATA_DIR) -> bool:
    """
    Download CMAPSS FD001 dataset from NASA prognostics data repository.
    Returns True if successful, False if offline.
    """
    data_dir.mkdir(exist_ok=True)
    train_file = data_dir / "train_FD001.txt"
    test_file  = data_dir / "test_FD001.txt"
    rul_file   = data_dir / "RUL_FD001.txt"

    if train_file.exists() and test_file.exists() and rul_file.exists():
        return True

    url = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
    try:
        print("Downloading NASA CMAPSS dataset...")
        with urllib.request.urlopen(url, timeout=15) as r:
            zip_data = r.read()
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            for name in z.namelist():
                if "FD001" in name:
                    z.extract(name, data_dir)
        print("Download complete.")
        return True
    except Exception:
        return False


def generate_synthetic_cmapss(n_units: int = 100, seed: int = 42,
                               data_dir: Path = DATA_DIR) -> None:
    """
    Generate synthetic CMAPSS-like data when offline.
    Simulates realistic turbofan engine degradation patterns.
    """
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []

    for unit in range(1, n_units + 1):
        max_cycle = int(rng.integers(150, 350))
        for cycle in range(1, max_cycle + 1):
            t = cycle / max_cycle          # normalized time 0→1
            deg = t ** 1.8                 # degradation curve (accelerating)

            # Operating conditions (3 clusters)
            op_mode = rng.integers(0, 3)
            op1 = [0.0, 0.42, 1.0][op_mode] + rng.normal(0, 0.01)
            op2 = [0.0, 14.0, 25.0][op_mode] + rng.normal(0, 0.1)
            op3 = [100.0, 84.0, 60.0][op_mode] + rng.normal(0, 0.2)

            # 14 useful sensor readings with realistic degradation
            sensors = [
                518.67 + rng.normal(0, 0.5),                   # s2  — total temp
                642.68 + 10 * deg + rng.normal(0, 0.5),        # s3  — total temp HPC
                1590.3 - 20 * deg + rng.normal(0, 3),          # s4  — total temp LPT
                14.62 + rng.normal(0, 0.01),                   # s7  — total pressure
                21.61 + 2 * deg + rng.normal(0, 0.2),          # s8  — fan speed
                554.36 + rng.normal(0, 1),                     # s9  — core speed
                2388.1 - 50 * deg + rng.normal(0, 5),          # s11 — HPC outlet pressure
                9065.4 - 100 * deg + rng.normal(0, 10),        # s12 — fan speed
                1.3 + 0.3 * deg + rng.normal(0, 0.01),         # s13 — corrected fan speed
                47.47 + 5 * deg + rng.normal(0, 0.3),          # s14 — corrected core speed
                521.66 + 3 * deg + rng.normal(0, 0.5),         # s15 — bypass ratio
                2388.1 - 30 * deg + rng.normal(0, 5),          # s17 — bleed enthalpy
                8138.6 - 80 * deg + rng.normal(0, 10),         # s20 — HPT coolant bleed
                8.4195 + 0.5 * deg + rng.normal(0, 0.05),      # s21 — LPT coolant bleed
            ]

            # Pad to 21 sensors (useless ones = constant)
            all_s = [np.nan] * 21
            useful_idx = [1,2,3,6,7,8,10,11,12,13,14,16,19,20]
            for i, idx in enumerate(useful_idx):
                all_s[idx] = sensors[i]
            # Fill useless with constants
            for j in [0,4,5,9,15,17,18]:
                all_s[j] = rng.normal(0, 0.0001)

            row = [unit, cycle, op1, op2, op3] + all_s
            rows.append(row)

    df = pd.DataFrame(rows, columns=ALL_COLS)
    df.to_csv(data_dir / "train_FD001.txt", sep=" ", index=False, header=False, na_rep="NaN")

    # Generate test set (shorter sequences — no failure)
    test_rows = []
    rul_list  = []
    for unit in range(1, 51):
        max_cycle = int(rng.integers(150, 350))
        obs_cycle = int(rng.integers(max_cycle // 2, max_cycle - 10))
        rul_list.append(max_cycle - obs_cycle)
        for cycle in range(1, obs_cycle + 1):
            t = cycle / max_cycle
            deg = t ** 1.8
            op_mode = rng.integers(0, 3)
            op1 = [0.0, 0.42, 1.0][op_mode] + rng.normal(0, 0.01)
            op2 = [0.0, 14.0, 25.0][op_mode] + rng.normal(0, 0.1)
            op3 = [100.0, 84.0, 60.0][op_mode] + rng.normal(0, 0.2)
            sensors = [
                518.67 + rng.normal(0, 0.5),
                642.68 + 10 * deg + rng.normal(0, 0.5),
                1590.3 - 20 * deg + rng.normal(0, 3),
                14.62 + rng.normal(0, 0.01),
                21.61 + 2 * deg + rng.normal(0, 0.2),
                554.36 + rng.normal(0, 1),
                2388.1 - 50 * deg + rng.normal(0, 5),
                9065.4 - 100 * deg + rng.normal(0, 10),
                1.3 + 0.3 * deg + rng.normal(0, 0.01),
                47.47 + 5 * deg + rng.normal(0, 0.3),
                521.66 + 3 * deg + rng.normal(0, 0.5),
                2388.1 - 30 * deg + rng.normal(0, 5),
                8138.6 - 80 * deg + rng.normal(0, 10),
                8.4195 + 0.5 * deg + rng.normal(0, 0.05),
            ]
            all_s = [np.nan] * 21
            useful_idx = [1,2,3,6,7,8,10,11,12,13,14,16,19,20]
            for i, idx in enumerate(useful_idx):
                all_s[idx] = sensors[i]
            for j in [0,4,5,9,15,17,18]:
                all_s[j] = rng.normal(0, 0.0001)
            test_rows.append([unit, cycle, op1, op2, op3] + all_s)

    test_df = pd.DataFrame(test_rows, columns=ALL_COLS)
    test_df.to_csv(data_dir / "test_FD001.txt", sep=" ", index=False, header=False, na_rep="NaN")
    rul_df = pd.DataFrame(rul_list)
    rul_df.to_csv(data_dir / "RUL_FD001.txt", index=False, header=False)
    print(f"Generated synthetic CMAPSS: {n_units} train units, 50 test units.")


def load_cmapss(data_dir: Path = DATA_DIR, subset: str = "FD001") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load CMAPSS train/test data. Returns (train_df, test_df, rul_series)."""
    def read_file(path):
        df = pd.read_csv(path, sep=r"\s+", header=None, names=ALL_COLS)
        df = df.dropna(how="all")
        return df

    train_df = read_file(data_dir / f"train_{subset}.txt")
    test_df  = read_file(data_dir / f"test_{subset}.txt")
    rul_df   = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None, names=["RUL"])
    return train_df, test_df, rul_df["RUL"]


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life column to training data."""
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def add_fault_label(df: pd.DataFrame, threshold: int = FAULT_CYCLES) -> pd.DataFrame:
    """Binary fault label: 1 if RUL <= threshold (in fault zone), 0 otherwise."""
    if "RUL" not in df.columns:
        df = add_rul(df)
    df["fault"] = (df["RUL"] <= threshold).astype(int)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract statistical features per unit per rolling window.
    Returns one row per (unit_id, cycle) with engineered features.
    """
    features = []
    sensor_cols = USEFUL_SENSORS

    for unit_id, group in df.groupby("unit_id"):
        group = group.sort_values("cycle").reset_index(drop=True)

        # Fill NaN sensors with forward fill
        group[sensor_cols] = group[sensor_cols].ffill().bfill().fillna(0)

        for i, row in group.iterrows():
            feat = {
                "unit_id": unit_id,
                "cycle":   row["cycle"],
            }
            # Add fault label if present
            if "fault" in group.columns:
                feat["fault"] = row["fault"]
            if "RUL" in group.columns:
                feat["RUL"] = row["RUL"]

            # Raw sensor values (normalized per unit later)
            for s in sensor_cols:
                feat[f"{s}_raw"] = row[s]

            # Rolling stats (window up to current point)
            window = group[sensor_cols].iloc[max(0, i-14):i+1]
            for s in sensor_cols:
                vals = window[s].values
                feat[f"{s}_mean"]  = float(np.mean(vals))
                feat[f"{s}_std"]   = float(np.std(vals) if len(vals) > 1 else 0)
                feat[f"{s}_slope"] = float(np.polyfit(range(len(vals)), vals, 1)[0]
                                           if len(vals) > 2 else 0)

            features.append(feat)

    return pd.DataFrame(features)


def build_dataset(data_dir: Path = DATA_DIR) -> Dict:
    """
    Full pipeline: load → feature extract → label → split → scale.
    Returns dict with X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    # Check if data exists, generate synthetic if not
    if not (data_dir / "train_FD001.txt").exists():
        ok = download_cmapss(data_dir)
        if not ok:
            print("Offline — generating synthetic CMAPSS data...")
            generate_synthetic_cmapss(data_dir=data_dir)

    train_raw, test_raw, test_rul = load_cmapss(data_dir)

    # Add labels
    train_raw = add_fault_label(train_raw)
    last_cycle = test_raw.groupby("unit_id")["cycle"].max()
    test_units = test_raw[test_raw["cycle"] == test_raw.groupby("unit_id")["cycle"].transform("max")].copy()

    # Extract features
    print("Extracting features from training data...")
    train_feat = extract_features(train_raw)
    print("Extracting features from test data...")
    test_feat  = extract_features(test_raw)

    # Feature columns (exclude metadata)
    meta_cols = ["unit_id", "cycle", "fault", "RUL"]
    feat_cols  = [c for c in train_feat.columns if c not in meta_cols]

    X_train = train_feat[feat_cols].fillna(0).values
    y_train = train_feat["fault"].values

    # For test: use last observation per unit + true RUL label
    test_last = test_feat.groupby("unit_id").last().reset_index()
    test_last["RUL_true"] = test_rul.values[:len(test_last)]
    test_last["fault"]    = (test_last["RUL_true"] <= FAULT_CYCLES).astype(int)
    X_test = test_last[feat_cols].fillna(0).values
    y_test = test_last["fault"].values

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"Dataset ready — Train: {X_train_sc.shape}, Test: {X_test_sc.shape}")
    print(f"Fault rate — Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    return {
        "X_train": X_train_sc,
        "X_test":  X_test_sc,
        "y_train": y_train,
        "y_test":  y_test,
        "scaler":  scaler,
        "feature_names": feat_cols,
        "train_feat": train_feat,
        "test_last":  test_last,
    }
