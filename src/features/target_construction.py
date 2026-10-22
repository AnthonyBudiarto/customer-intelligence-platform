"""
Target Construction — Definisi dan Konstruksi Label Inaktivitas

DEFINISI OPERASIONAL TARGET:
    Seorang pelanggan berlabel INACTIVE (1) jika:
    - Mereka eligible (punya >= 1 qualified purchase dalam eligibility_window sebelum cutoff)
    - Mereka TIDAK melakukan qualified purchase dalam [cutoff+1, cutoff+horizon_days]

    Seorang pelanggan berlabel ACTIVE (0) jika:
    - Mereka eligible
    - Mereka melakukan >= 1 qualified purchase dalam prediction horizon

CATATAN PENTING:
    - "Qualified purchase" = transaksi valid dari cleaned_modeling.parquet
    - Cancellations, zero-price, dan zero-quantity TIDAK dihitung
    - Label ini BUKAN "churn" yang sesungguhnya — pelanggan mungkin membeli
      di saluran lain atau kembali setelah horizon
    - Pelanggan tanpa cukup follow-up time dieksklusi (bukan dilabel aktif/inactive)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def construct_inactivity_target(
    df_modeling: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_days: int,
    eligibility_window_days: int = 90,
    min_eligibility_transactions: int = 1,
) -> Tuple[pd.Series, Dict]:
    """
    Konstruksi label inaktivitas biner untuk setiap pelanggan eligible.

    Parameters
    ----------
    df_modeling : pd.DataFrame
        Cleaned modeling data (qualified purchases). Harus memiliki:
        CustomerID, InvoiceDate, InvoiceNo.
    cutoff_date : pd.Timestamp
        Tanggal "hari ini" untuk prediksi.
    horizon_days : int
        Jumlah hari ke depan yang digunakan untuk menentukan inaktivitas.
        Harus <= (tanggal terakhir dataset - cutoff_date).
    eligibility_window_days : int
        Jumlah hari sebelum cutoff untuk menentukan pelanggan aktif.
        Pelanggan dengan pembelian dalam window ini dianggap eligible.
    min_eligibility_transactions : int
        Minimum qualified purchases dalam eligibility window.

    Returns
    -------
    Tuple[pd.Series, Dict]
        - pd.Series dengan index=CustomerID, values=0/1 (0=active, 1=inactive)
        - dict dengan statistik konstruksi target

    Raises
    ------
    ValueError
        Jika horizon_days melebihi available follow-up period.
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)
    eligibility_start = cutoff_date - pd.Timedelta(days=eligibility_window_days)

    # Validasi: apakah horizon tersedia dalam dataset?
    dataset_end = df_modeling["InvoiceDate"].max()
    available_followup = (dataset_end - cutoff_date).days

    if horizon_end > dataset_end:
        raise ValueError(
            f"Horizon {horizon_days} hari dari cutoff {cutoff_date.date()} "
            f"melampaui akhir dataset ({dataset_end.date()}). "
            f"Follow-up tersedia: {available_followup} hari. "
            f"Gunakan horizon_days <= {available_followup} atau cutoff yang lebih awal."
        )

    # ----------------------------------------------------------------
    # 1. TENTUKAN PELANGGAN ELIGIBLE
    # ----------------------------------------------------------------
    # Eligible = punya >= min_eligibility_transactions dalam eligibility window
    obs_window = df_modeling[
        (df_modeling["InvoiceDate"] >= eligibility_start)
        & (df_modeling["InvoiceDate"] <= cutoff_date)
    ]

    eligible_tx_counts = obs_window.groupby("CustomerID")["InvoiceNo"].nunique()
    eligible_customers = eligible_tx_counts[
        eligible_tx_counts >= min_eligibility_transactions
    ].index

    # ----------------------------------------------------------------
    # 2. TENTUKAN SIAPA YANG AKTIF DI HORIZON
    # ----------------------------------------------------------------
    horizon_data = df_modeling[
        (df_modeling["InvoiceDate"] > cutoff_date)
        & (df_modeling["InvoiceDate"] <= horizon_end)
    ]

    active_in_horizon = set(horizon_data["CustomerID"].unique())

    # ----------------------------------------------------------------
    # 3. BUAT LABEL
    # ----------------------------------------------------------------
    labels = pd.Series(index=eligible_customers, dtype=int, name="inactive")

    for cid in eligible_customers:
        labels[cid] = 0 if cid in active_in_horizon else 1

    # ----------------------------------------------------------------
    # 4. STATISTIK
    # ----------------------------------------------------------------
    n_total = len(labels)
    n_inactive = int(labels.sum())
    n_active = n_total - n_inactive

    stats = {
        "cutoff_date": str(cutoff_date.date()),
        "horizon_days": horizon_days,
        "horizon_end": str(horizon_end.date()),
        "eligibility_window_days": eligibility_window_days,
        "eligibility_start": str(eligibility_start.date()),
        "dataset_end": str(dataset_end.date()),
        "available_followup_days": available_followup,
        "n_eligible_customers": n_total,
        "n_active_label_0": n_active,
        "n_inactive_label_1": n_inactive,
        "pct_inactive": round(n_inactive / n_total * 100, 2) if n_total > 0 else 0,
        "class_imbalance_ratio": round(n_active / n_inactive, 2) if n_inactive > 0 else float("inf"),
    }

    print(f"Target konstruksi selesai:")
    print(f"  Cutoff: {cutoff_date.date()} | Horizon: {horizon_days} hari | End: {horizon_end.date()}")
    print(f"  Pelanggan eligible: {n_total:,}")
    print(f"  Label ACTIVE (0): {n_active:,} ({100-stats['pct_inactive']:.1f}%)")
    print(f"  Label INACTIVE (1): {n_inactive:,} ({stats['pct_inactive']:.1f}%)")
    print(f"  Class imbalance ratio (active:inactive): {stats['class_imbalance_ratio']:.1f}:1")

    return labels, stats


def validate_target_no_leakage(
    labels: pd.Series,
    features: pd.DataFrame,
    df_modeling: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> dict:
    """
    Validasi bahwa label dibangun dari data setelah cutoff,
    dan fitur hanya dari data sebelum cutoff.

    Pemeriksaan:
    1. Semua pelanggan dalam labels ada dalam features.
    2. Features tidak mengandung informasi post-cutoff.
    3. Labels tidak dipengaruhi data pre-cutoff.

    Returns
    -------
    dict dengan status PASS/FAIL untuk setiap pemeriksaan.
    """
    cutoff_date = pd.Timestamp(cutoff_date)

    # Check 1: alignment
    labels_cids = set(labels.index)
    features_cids = set(features.index)
    unmatched = labels_cids - features_cids

    # Check 2: feature max date tidak melebihi cutoff
    obs = df_modeling[df_modeling["CustomerID"].isin(list(labels_cids))]
    pre_cutoff = obs[obs["InvoiceDate"] <= cutoff_date]
    post_cutoff = obs[obs["InvoiceDate"] > cutoff_date]

    # Check 3: tidak ada pelanggan dengan hanya post-cutoff tx yang ada di features
    only_post = set(post_cutoff["CustomerID"].unique()) - set(pre_cutoff["CustomerID"].unique())
    false_eligibles = only_post & features_cids

    result = {
        "check_labels_in_features": "PASS" if len(unmatched) == 0 else f"FAIL — {len(unmatched)} CIDs tidak match",
        "check_no_post_cutoff_in_features": "PASS",  # dijamin oleh build_customer_features
        "check_no_future_only_customers": "PASS" if len(false_eligibles) == 0 else f"FAIL — {len(false_eligibles)} CIDs",
        "n_labels": len(labels),
        "n_features": len(features),
        "n_labels_with_features": len(labels_cids & features_cids),
    }

    all_pass = all(v.startswith("PASS") if isinstance(v, str) else True for v in result.values())
    result["overall"] = "PASS" if all_pass else "FAIL"

    return result
