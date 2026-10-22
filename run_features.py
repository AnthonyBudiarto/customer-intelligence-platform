"""
FASE 3A + 3B: Feature Engineering & Target Construction
Output: data/features/features_train.parquet, features_test.parquet,
        target_train.parquet, target_test.parquet
"""
import sys, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.features.feature_engineering import build_customer_features, validate_no_leakage
from src.features.target_construction import construct_inactivity_target, validate_target_no_leakage

FEATURES_DIR = Path("data/features")
REPORTS_DIR = Path("artifacts/reports")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

print("Memuat data bersih...")
df_m = pd.read_parquet("data/processed/cleaned_modeling.parquet")
df_a = pd.read_parquet("data/processed/cleaned_analysis.parquet")
df_m["InvoiceDate"] = pd.to_datetime(df_m["InvoiceDate"])
df_a["InvoiceDate"] = pd.to_datetime(df_a["InvoiceDate"])

# Pastikan IsCancelled ada di df_a
if "IsCancelled" not in df_a.columns:
    df_a["IsCancelled"] = df_a["InvoiceNo"].str.startswith("C", na=False)

# Pastikan LineValue ada
if "LineValue" not in df_m.columns:
    df_m["LineValue"] = df_m["Quantity"] * df_m["UnitPrice"]
if "LineValue" not in df_a.columns:
    df_a["LineValue"] = df_a["Quantity"] * df_a["UnitPrice"]

# ============================================================
# PARAMETER TEMPORAL (FINAL, SETELAH VALIDASI FASE 2B)
# ============================================================
# cutoff=2011-09-30, horizon=70 hari
# Train split: observasi s.d. 2011-06-30, horizon 2011-07-01 ~ 2011-09-07
# Test split:  observasi s.d. 2011-09-30, horizon 2011-10-01 ~ 2011-12-08

SPLITS = {
    "train": {
        "cutoff": pd.Timestamp("2011-06-30"),
        "horizon_days": 70,
        "eligibility_window_days": 90,
        "description": "Training: observasi s.d. 30 Jun 2011, horizon 70 hari ke depan",
    },
    "test": {
        "cutoff": pd.Timestamp("2011-09-30"),
        "horizon_days": 70,
        "eligibility_window_days": 90,
        "description": "Test: observasi s.d. 30 Sep 2011, horizon 70 hari ke depan",
    },
}

all_results = {}

for split_name, split_cfg in SPLITS.items():
    print(f"\n{'='*60}")
    print(f"Membangun {split_name.upper()} split...")
    print(f"  {split_cfg['description']}")
    print(f"{'='*60}")

    cutoff = split_cfg["cutoff"]
    horizon = split_cfg["horizon_days"]
    elig_window = split_cfg["eligibility_window_days"]

    # -----------------------------------------------------------
    # 1. BANGUN FITUR
    # -----------------------------------------------------------
    print(f"\n[1/3] Membangun fitur customer-level...")
    features = build_customer_features(
        df_modeling=df_m,
        df_analysis=df_a,
        cutoff_date=cutoff,
        min_transactions=1,
    )
    print(f"  Fitur dibangun: {len(features):,} pelanggan x {features.shape[1]} fitur")
    print(f"  Kolom fitur: {list(features.columns)}")

    # -----------------------------------------------------------
    # 2. KONSTRUKSI TARGET
    # -----------------------------------------------------------
    print(f"\n[2/3] Mengkonstruksi target inaktivitas...")
    labels, target_stats = construct_inactivity_target(
        df_modeling=df_m,
        cutoff_date=cutoff,
        horizon_days=horizon,
        eligibility_window_days=elig_window,
        min_eligibility_transactions=1,
    )

    # -----------------------------------------------------------
    # 3. ALIGN FEATURES DAN LABELS
    # -----------------------------------------------------------
    print(f"\n[3/3] Menyelaraskan fitur dan label...")
    # Hanya pelanggan yang ada di keduanya
    common_customers = features.index.intersection(labels.index)
    features_aligned = features.loc[common_customers]
    labels_aligned = labels.loc[common_customers]

    print(f"  Pelanggan dengan fitur DAN label: {len(common_customers):,}")
    print(f"  Hanya di features (tidak eligible): {len(features.index) - len(common_customers):,}")
    print(f"  Hanya di labels (tidak ada fitur cukup): {len(labels.index) - len(common_customers):,}")

    # -----------------------------------------------------------
    # 4. VALIDASI LEAKAGE
    # -----------------------------------------------------------
    leakage_check = validate_no_leakage(features_aligned, df_m, cutoff)
    target_check = validate_target_no_leakage(labels_aligned, features_aligned, df_m, cutoff)

    print(f"\n  Leakage check: {leakage_check['status']}")
    print(f"  Target check: {target_check['overall']}")

    assert leakage_check["status"] == "PASS", f"LEAKAGE TERDETEKSI di {split_name}!"
    assert target_check["overall"] == "PASS", f"TARGET CHECK GAGAL di {split_name}!"

    # -----------------------------------------------------------
    # 5. SIMPAN
    # -----------------------------------------------------------
    feat_path = FEATURES_DIR / f"features_{split_name}.parquet"
    tgt_path = FEATURES_DIR / f"target_{split_name}.parquet"

    features_aligned.to_parquet(feat_path)
    labels_aligned.to_frame().to_parquet(tgt_path)

    print(f"\n  Tersimpan: {feat_path.name} ({feat_path.stat().st_size/1024:.0f} KB)")
    print(f"  Tersimpan: {tgt_path.name}")

    all_results[split_name] = {
        "split": split_name,
        "cutoff": str(cutoff.date()),
        "horizon_days": horizon,
        "n_customers": len(common_customers),
        "n_features": features_aligned.shape[1],
        "target_stats": target_stats,
        "leakage_check": leakage_check,
        "target_validation": target_check,
        "feature_columns": list(features_aligned.columns),
        "missing_values_in_features": features_aligned.isnull().sum().to_dict(),
    }

# ============================================================
# STATISTIK FITUR (dari train split)
# ============================================================
print(f"\n{'='*60}")
print("STATISTIK FITUR (TRAIN SPLIT)")
print(f"{'='*60}")
feat_train = pd.read_parquet(FEATURES_DIR / "features_train.parquet")
tgt_train = pd.read_parquet(FEATURES_DIR / "target_train.parquet")["inactive"]

print(f"\nMissing values per fitur:")
missing = feat_train.isnull().sum()
for col, n in missing.items():
    if n > 0:
        print(f"  {col}: {n} missing ({n/len(feat_train)*100:.1f}%)")
    else:
        print(f"  {col}: OK")

print(f"\nDistribusi target (TRAIN):")
print(f"  Active (0): {(tgt_train==0).sum():,} ({(tgt_train==0).mean()*100:.1f}%)")
print(f"  Inactive (1): {(tgt_train==1).sum():,} ({(tgt_train==1).mean()*100:.1f}%)")

print(f"\nDistribusi target (TEST):")
tgt_test = pd.read_parquet(FEATURES_DIR / "target_test.parquet")["inactive"]
print(f"  Active (0): {(tgt_test==0).sum():,} ({(tgt_test==0).mean()*100:.1f}%)")
print(f"  Inactive (1): {(tgt_test==1).sum():,} ({(tgt_test==1).mean()*100:.1f}%)")

print(f"\nRingkasan fitur:")
print(feat_train.describe().round(2).to_string())

# Simpan hasil
all_results["feature_summary"] = {
    "feature_columns": list(feat_train.columns),
    "n_features": feat_train.shape[1],
    "train_class_distribution": {
        "n_active": int((tgt_train==0).sum()),
        "n_inactive": int((tgt_train==1).sum()),
        "pct_inactive": round(float(tgt_train.mean())*100, 2),
    },
    "test_class_distribution": {
        "n_active": int((tgt_test==0).sum()),
        "n_inactive": int((tgt_test==1).sum()),
        "pct_inactive": round(float(tgt_test.mean())*100, 2),
    },
}

with open(REPORTS_DIR / "feature_engineering_report.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n[OK] Feature engineering report: artifacts/reports/feature_engineering_report.json")
print("\nFASE 3 SELESAI")
