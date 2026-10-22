"""
FASE 4A: Customer Segmentation — RFM + K-Means
Segmentasi dilakukan pada data per cutoff TEST (2011-09-30) agar
representatif untuk dashboard. Scaler difit hanya pada data ini.
"""
import sys, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation.rfm_segmentation import (
    compute_rfm, transform_rfm, fit_clustering, profile_clusters
)

PLOTS_DIR = Path("artifacts/plots")
REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
FEATURES_DIR = Path("data/features")
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

print("Memuat data...")
df_m = pd.read_parquet("data/processed/cleaned_modeling.parquet")
df_m["InvoiceDate"] = pd.to_datetime(df_m["InvoiceDate"])
if "LineValue" not in df_m.columns:
    df_m["LineValue"] = df_m["Quantity"] * df_m["UnitPrice"]

CUTOFF = pd.Timestamp("2011-09-30")

# ============================================================
# 1. HITUNG RFM
# ============================================================
print("\n[1/5] Menghitung RFM...")
rfm = compute_rfm(df_m, cutoff_date=CUTOFF)
print(f"  RFM dihitung untuk {len(rfm):,} pelanggan")
print(f"  Recency  — median: {rfm['recency_days'].median():.0f} hari, max: {rfm['recency_days'].max():.0f} hari")
print(f"  Frequency — median: {rfm['frequency'].median():.0f}, max: {rfm['frequency'].max():.0f}")
print(f"  Monetary — median: £{rfm['monetary_total'].median():.0f}, max: £{rfm['monetary_total'].max():,.0f}")

# ============================================================
# 2. TRANSFORMASI LOG
# ============================================================
print("\n[2/5] Transformasi log...")
rfm_log, transform_stats = transform_rfm(rfm)
print(f"  Frequency skew sebelum log: {transform_stats['frequency_original_skew']:.2f}")
print(f"  Frequency skew sesudah log: {transform_stats['frequency_log_skew']:.2f}")
print(f"  Monetary  skew sebelum log: {transform_stats['monetary_original_skew']:.2f}")
print(f"  Monetary  skew sesudah log: {transform_stats['monetary_log_skew']:.2f}")

# ============================================================
# 3. CLUSTERING
# ============================================================
print("\n[3/5] K-Means clustering (evaluasi k=2..8)...")
scaler, best_km, cluster_labels, sil_df = fit_clustering(
    rfm_log, k_range=range(2, 9), random_state=42
)

# ============================================================
# 4. PROFIL SEGMEN
# ============================================================
print("\n[4/5] Profiling segmen...")
profile = profile_clusters(rfm, cluster_labels)

# Gabungkan RFM + cluster + segment name
rfm_clustered = rfm.join(cluster_labels).copy()
rfm_clustered = rfm_clustered.join(
    profile[["segment_name"]].rename(columns={"segment_name": "segment_name_ref"}),
    on="cluster"
)
rfm_clustered["segment_name"] = rfm_clustered["segment_name_ref"]
rfm_clustered.drop("segment_name_ref", axis=1, inplace=True)
rfm_clustered["cluster_label"] = rfm_clustered["cluster"].map(
    profile["segment_name"].to_dict()
)

# ============================================================
# 5. VISUALISASI
# ============================================================
print("\n[5/5] Membuat visualisasi segmentasi...")
n_clusters = best_km.n_clusters
SEGMENT_COLORS = sns.color_palette("Set2", n_clusters)
cluster_color_map = {i: SEGMENT_COLORS[i] for i in range(n_clusters)}

# --- Plot A: Silhouette scores ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["green" if k == best_km.n_clusters else "steelblue" for k in sil_df["k"]]
bars = ax.bar(sil_df["k"], sil_df["silhouette_score"], color=colors, alpha=0.8, edgecolor="white")
ax.set_title("Silhouette Score vs Jumlah Cluster\n(hijau = k optimal)", fontweight="bold")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Silhouette Score")
for bar, val in zip(bars, sil_df["silhouette_score"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "11_silhouette_scores.png", dpi=120, bbox_inches="tight")
plt.close()

# --- Plot B: Radar chart untuk profil segmen ---
from matplotlib.patches import FancyArrowPatch
categories = ["Recency\n(rendah=baik)", "Frequency\n(tinggi=baik)", "Monetary\n(tinggi=baik)"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5), subplot_kw=dict(polar=True))
if n_clusters == 1:
    axes = [axes]

# Normalisasi metrik untuk radar (0-1 scale berdasarkan min-max di semua cluster)
r_max = profile["recency_median"].max()
r_min = profile["recency_median"].min()
f_max = profile["frequency_median"].max()
f_min = profile["frequency_median"].min()
m_max = profile["monetary_median"].max()
m_min = profile["monetary_median"].min()

for i, (cluster_id, row) in enumerate(profile.iterrows()):
    ax = axes[i]
    # Recency dibalik (rendah = lebih baik -> nilai tinggi di radar)
    if r_max > r_min:
        rec_norm = 1 - (row["recency_median"] - r_min) / (r_max - r_min)
    else:
        rec_norm = 0.5
    freq_norm = (row["frequency_median"] - f_min) / (f_max - f_min) if f_max > f_min else 0.5
    mon_norm = (row["monetary_median"] - m_min) / (m_max - m_min) if m_max > m_min else 0.5

    values = [rec_norm, freq_norm, mon_norm]
    values += values[:1]

    ax.plot(angles, values, "o-", linewidth=2, color=SEGMENT_COLORS[i])
    ax.fill(angles, values, alpha=0.25, color=SEGMENT_COLORS[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"Cluster {cluster_id}\n{row['segment_name']}\n(n={int(row['n_customers']):,})",
                 size=10, fontweight="bold", pad=15)

plt.suptitle("Profil Segmen Pelanggan (Radar Chart)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "12_segment_radar.png", dpi=120, bbox_inches="tight")
plt.close()

# --- Plot C: Box plots RFM per cluster ---
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
rfm_plot = rfm_clustered.copy()
rfm_plot["cluster_label"] = rfm_plot["cluster"].map(profile["segment_name"].to_dict())

axes[0].boxplot(
    [rfm_plot[rfm_plot["cluster"]==c]["recency_days"].values for c in sorted(rfm_plot["cluster"].unique())],
    labels=[f"C{c}\n{profile.loc[c,'segment_name'][:10]}" for c in sorted(rfm_plot["cluster"].unique())],
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
)
axes[0].set_title("Recency per Cluster", fontweight="bold")
axes[0].set_ylabel("Hari Sejak Pembelian Terakhir")
axes[0].tick_params(axis="x", labelsize=8)

axes[1].boxplot(
    [np.log1p(rfm_plot[rfm_plot["cluster"]==c]["frequency"].values) for c in sorted(rfm_plot["cluster"].unique())],
    labels=[f"C{c}\n{profile.loc[c,'segment_name'][:10]}" for c in sorted(rfm_plot["cluster"].unique())],
    patch_artist=True,
    boxprops=dict(facecolor="lightgreen"),
)
axes[1].set_title("Frequency per Cluster (log scale)", fontweight="bold")
axes[1].set_ylabel("log(1 + Frequency)")
axes[1].tick_params(axis="x", labelsize=8)

axes[2].boxplot(
    [np.log1p(rfm_plot[rfm_plot["cluster"]==c]["monetary_total"].values) for c in sorted(rfm_plot["cluster"].unique())],
    labels=[f"C{c}\n{profile.loc[c,'segment_name'][:10]}" for c in sorted(rfm_plot["cluster"].unique())],
    patch_artist=True,
    boxprops=dict(facecolor="lightsalmon"),
)
axes[2].set_title("Monetary per Cluster (log scale)", fontweight="bold")
axes[2].set_ylabel("log(1 + Monetary Total)")
axes[2].tick_params(axis="x", labelsize=8)

plt.suptitle("Distribusi RFM per Segmen", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "13_rfm_boxplots.png", dpi=120, bbox_inches="tight")
plt.close()

# --- Plot D: Ukuran dan revenue per segmen ---
seg_revenue = rfm_clustered.groupby("cluster_label").agg(
    n_customers=("monetary_total", "count"),
    total_revenue=("monetary_total", "sum"),
    avg_revenue=("monetary_total", "mean"),
).reset_index()
seg_revenue["revenue_share_pct"] = seg_revenue["total_revenue"] / seg_revenue["total_revenue"].sum() * 100
seg_revenue["customer_share_pct"] = seg_revenue["n_customers"] / seg_revenue["n_customers"].sum() * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
colors_pie = sns.color_palette("Set2", len(seg_revenue))

axes[0].pie(seg_revenue["n_customers"], labels=seg_revenue["cluster_label"],
            autopct="%1.1f%%", colors=colors_pie, startangle=90)
axes[0].set_title("Distribusi Pelanggan per Segmen", fontweight="bold")

axes[1].pie(seg_revenue["total_revenue"], labels=seg_revenue["cluster_label"],
            autopct="%1.1f%%", colors=colors_pie, startangle=90)
axes[1].set_title("Distribusi Revenue per Segmen", fontweight="bold")

plt.suptitle("Ukuran dan Revenue per Segmen Pelanggan", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "14_segment_size_revenue.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================
# SIMPAN MODEL DAN DATA
# ============================================================
joblib.dump(scaler, MODELS_DIR / "rfm_scaler.joblib")
joblib.dump(best_km, MODELS_DIR / "kmeans_model.joblib")

rfm_clustered.to_parquet(FEATURES_DIR / "rfm_segments.parquet")
profile.to_csv(REPORTS_DIR / "segment_profiles.csv")

seg_report = {
    "cutoff_date": str(CUTOFF.date()),
    "n_customers_segmented": int(len(rfm_clustered)),
    "best_k": int(best_km.n_clusters),
    "best_silhouette": float(sil_df.loc[sil_df["silhouette_score"].idxmax(), "silhouette_score"]),
    "silhouette_scores": sil_df.to_dict("records"),
    "transform_stats": transform_stats,
    "segment_profiles": profile.reset_index().to_dict("records"),
    "segment_revenue_summary": seg_revenue.to_dict("records"),
}
with open(REPORTS_DIR / "segmentation_report.json", "w") as f:
    json.dump(seg_report, f, indent=2, default=str)

print(f"\nModel tersimpan: artifacts/models/rfm_scaler.joblib")
print(f"Model tersimpan: artifacts/models/kmeans_model.joblib")
print(f"Segmen tersimpan: data/features/rfm_segments.parquet")
print(f"Report tersimpan: artifacts/reports/segmentation_report.json")
print("\nFASE 4A SELESAI")
