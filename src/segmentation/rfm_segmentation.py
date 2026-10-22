"""
Customer Segmentation — RFM Analysis dan K-Means Clustering

Alur:
1. Hitung RFM dari data observasi (sebelum cutoff)
2. Transformasi log dan standardisasi (fit HANYA pada data yang relevan)
3. Evaluasi cluster dengan silhouette score untuk k=2..10
4. Fit K-Means dengan k optimal
5. Profil dan namai segmen berdasarkan karakteristik terukur
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


def compute_rfm(
    df_modeling: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Hitung fitur RFM untuk semua pelanggan hingga cutoff_date.

    Parameters
    ----------
    df_modeling : pd.DataFrame
        Cleaned modeling data.
    cutoff_date : pd.Timestamp
        Tanggal referensi untuk menghitung recency.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom: recency_days, frequency, monetary_total
        Index: CustomerID
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    obs = df_modeling[df_modeling["InvoiceDate"] <= cutoff_date].copy()

    if "LineValue" not in obs.columns:
        obs["LineValue"] = obs["Quantity"] * obs["UnitPrice"]

    # Recency
    last_purchase = obs.groupby("CustomerID")["InvoiceDate"].max()
    recency = (cutoff_date - last_purchase).dt.days.rename("recency_days")

    # Frequency
    frequency = obs.groupby("CustomerID")["InvoiceNo"].nunique().rename("frequency")

    # Monetary
    monetary = obs.groupby("CustomerID")["LineValue"].sum().rename("monetary_total")

    rfm = pd.concat([recency, frequency, monetary], axis=1)
    rfm.index.name = "CustomerID"
    return rfm


def transform_rfm(rfm: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Terapkan log1p transform untuk menangani skewness.
    Recency tidak di-log karena interpretasinya berbeda.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        - DataFrame terTransformasi
        - dict berisi statistik transformasi (untuk digunakan di inference)
    """
    rfm_t = rfm.copy()

    # Log transform untuk frequency dan monetary (sangat skewed)
    rfm_t["frequency_log"] = np.log1p(rfm_t["frequency"])
    rfm_t["monetary_log"] = np.log1p(rfm_t["monetary_total"])

    # Recency dibalik: pelanggan dengan recency rendah (baru saja beli) = lebih bernilai
    # Kita gunakan recency langsung (bukan dibalik) agar interpretasinya intuitif
    rfm_t["recency_log"] = np.log1p(rfm_t["recency_days"])

    transform_stats = {
        "frequency_original_skew": float(rfm["frequency"].skew()),
        "monetary_original_skew": float(rfm["monetary_total"].skew()),
        "frequency_log_skew": float(rfm_t["frequency_log"].skew()),
        "monetary_log_skew": float(rfm_t["monetary_log"].skew()),
    }

    return rfm_t[["recency_log", "frequency_log", "monetary_log"]], transform_stats


def fit_clustering(
    rfm_transformed: pd.DataFrame,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> Tuple[object, object, pd.Series, pd.DataFrame]:
    """
    Fit StandardScaler dan K-Means, evaluasi k dengan silhouette score.

    Parameters
    ----------
    rfm_transformed : pd.DataFrame
        Output dari transform_rfm().
    k_range : range
        Range nilai k yang akan dievaluasi.
    random_state : int
        Seed untuk reprodusibilitas.

    Returns
    -------
    Tuple berisi:
        - scaler (StandardScaler yang sudah difit)
        - best_kmeans (KMeans yang sudah difit dengan k optimal)
        - cluster_labels (pd.Series)
        - silhouette_scores (pd.DataFrame dengan k dan score)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Fit scaler HANYA pada data ini (training data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_transformed)

    # Evaluasi silhouette untuk berbagai k
    sil_scores = []
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append({"k": k, "silhouette_score": round(sil, 4), "inertia": km.inertia_})
        inertias.append(km.inertia_)

    sil_df = pd.DataFrame(sil_scores)
    best_k = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])

    print(f"\nSilhouette scores:")
    for _, row in sil_df.iterrows():
        marker = " <-- OPTIMAL" if row["k"] == best_k else ""
        print(f"  k={int(row['k'])}: silhouette={row['silhouette_score']:.4f}{marker}")

    # Fit best K-Means
    best_km = KMeans(n_clusters=best_k, n_init=10, max_iter=300, random_state=random_state)
    cluster_labels = best_km.fit_predict(X_scaled)
    cluster_series = pd.Series(cluster_labels, index=rfm_transformed.index, name="cluster")

    print(f"\nCluster terpilih: k={best_k}")
    print(f"Distribusi cluster:")
    for c, n in cluster_series.value_counts().sort_index().items():
        print(f"  Cluster {c}: {n:,} pelanggan ({n/len(cluster_series)*100:.1f}%)")

    return scaler, best_km, cluster_series, sil_df


def profile_clusters(
    rfm: pd.DataFrame,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    """
    Buat profil statistik untuk setiap cluster.

    Returns
    -------
    pd.DataFrame
        Profil dengan median RFM per cluster + nama segmen.
    """
    combined = rfm.join(cluster_labels)

    profile = combined.groupby("cluster").agg(
        n_customers=("recency_days", "count"),
        recency_median=("recency_days", "median"),
        recency_mean=("recency_days", "mean"),
        frequency_median=("frequency", "median"),
        frequency_mean=("frequency", "mean"),
        monetary_median=("monetary_total", "median"),
        monetary_mean=("monetary_total", "mean"),
    ).round(2)

    # Beri nama segmen berdasarkan karakteristik terukur
    # Logika: bandingkan setiap cluster dengan median keseluruhan
    overall_recency_med = rfm["recency_days"].median()
    overall_freq_med = rfm["frequency"].median()
    overall_monetary_med = rfm["monetary_total"].median()

    segment_names = {}
    for cluster_id, row in profile.iterrows():
        is_recent = row["recency_median"] <= overall_recency_med
        is_frequent = row["frequency_median"] >= overall_freq_med
        is_high_value = row["monetary_median"] >= overall_monetary_med

        if is_recent and is_frequent and is_high_value:
            name = "Champions"
        elif is_recent and is_frequent and not is_high_value:
            name = "Loyal Buyers"
        elif is_recent and not is_frequent and is_high_value:
            name = "High-Value Recent"
        elif is_recent and not is_frequent and not is_high_value:
            name = "New / Occasional"
        elif not is_recent and is_frequent and is_high_value:
            name = "At-Risk High Value"
        elif not is_recent and is_frequent and not is_high_value:
            name = "At-Risk Frequent"
        elif not is_recent and not is_frequent and is_high_value:
            name = "Lapsed Big Spenders"
        else:
            name = "Inactive Low Value"

        segment_names[cluster_id] = name

    profile["segment_name"] = pd.Series(segment_names)
    profile["pct_customers"] = (profile["n_customers"] / profile["n_customers"].sum() * 100).round(2)

    print(f"\nProfil Segmen:")
    print(f"{'Cluster':<8} {'Nama':<25} {'N':>6} {'%':>6} {'Rec_med':>8} {'Freq_med':>9} {'Mon_med':>10}")
    print("-" * 80)
    for cid, row in profile.iterrows():
        print(f"  {cid:<6} {row['segment_name']:<25} {int(row['n_customers']):>6,} "
              f"{row['pct_customers']:>5.1f}% {row['recency_median']:>8.0f} "
              f"{row['frequency_median']:>9.0f} {row['monetary_median']:>10.0f}")

    return profile
