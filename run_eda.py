"""
FASE 2A + 2B: EDA dan Temporal Design Validation
Semua output disimpan ke artifacts/plots/ dan artifacts/reports/
Tidak ada data masa depan yang digunakan.
"""
import sys, os, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

PLOTS_DIR = Path("artifacts/plots")
REPORTS_DIR = Path("artifacts/reports")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted")

print("Memuat data...")
df_a = pd.read_parquet("data/processed/cleaned_analysis.parquet")
df_m = pd.read_parquet("data/processed/cleaned_modeling.parquet")

# Pastikan InvoiceDate adalah datetime
df_a["InvoiceDate"] = pd.to_datetime(df_a["InvoiceDate"])
df_m["InvoiceDate"] = pd.to_datetime(df_m["InvoiceDate"])

eda_stats = {}

# ============================================================
# 1. REVENUE BULANAN
# ============================================================
print("Plot 1: Revenue bulanan...")
df_m["YM"] = df_m["InvoiceDate"].dt.to_period("M")
monthly_rev = df_m.groupby("YM")["LineValue"].sum().reset_index()
monthly_rev["YM_str"] = monthly_rev["YM"].astype(str)
monthly_orders = df_m.groupby("YM")["InvoiceNo"].nunique().reset_index()
monthly_orders.columns = ["YM", "n_orders"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
axes[0].bar(monthly_rev["YM_str"], monthly_rev["LineValue"] / 1000, color=COLORS[0], alpha=0.8)
axes[0].set_title("Revenue Bulanan (Cleaned Modeling Data)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Revenue (GBP Thousands)")
axes[0].tick_params(axis="x", rotation=45)
for i, v in enumerate(monthly_rev["LineValue"] / 1000):
    axes[0].text(i, v + 5, f"£{v:.0f}k", ha="center", va="bottom", fontsize=7)

monthly_orders_str = monthly_orders.copy()
monthly_orders_str["YM_str"] = monthly_orders_str["YM"].astype(str)
axes[1].bar(monthly_orders_str["YM_str"], monthly_orders_str["n_orders"], color=COLORS[1], alpha=0.8)
axes[1].set_title("Jumlah Order Unik per Bulan", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Jumlah Invoice Unik")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_monthly_revenue_orders.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["monthly_revenue"] = {
    "peak_month": str(monthly_rev.loc[monthly_rev["LineValue"].idxmax(), "YM"]),
    "peak_revenue": round(float(monthly_rev["LineValue"].max()), 2),
    "lowest_month": str(monthly_rev.loc[monthly_rev["LineValue"].idxmin(), "YM"]),
    "lowest_revenue": round(float(monthly_rev["LineValue"].min()), 2),
    "total_revenue": round(float(monthly_rev["LineValue"].sum()), 2),
}
print(f"  Peak bulan: {eda_stats['monthly_revenue']['peak_month']} = GBP {eda_stats['monthly_revenue']['peak_revenue']:,.0f}")

# ============================================================
# 2. DISTRIBUSI FREKUENSI TRANSAKSI PER PELANGGAN
# ============================================================
print("Plot 2: Distribusi frekuensi per pelanggan...")
cust_freq = df_m.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
cust_freq.columns = ["CustomerID", "n_invoices"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histogram (capped at 30)
capped = cust_freq["n_invoices"].clip(upper=30)
axes[0].hist(capped, bins=30, color=COLORS[2], alpha=0.8, edgecolor="white")
axes[0].set_title("Distribusi Frekuensi Transaksi\n(dikap pada 30 untuk visualisasi)", fontweight="bold")
axes[0].set_xlabel("Jumlah Invoice Unik")
axes[0].set_ylabel("Jumlah Pelanggan")
axes[0].axvline(cust_freq["n_invoices"].median(), color="red", linestyle="--", label=f"Median={cust_freq['n_invoices'].median():.0f}")
axes[0].legend()

# Log scale
axes[1].hist(np.log1p(cust_freq["n_invoices"]), bins=40, color=COLORS[3], alpha=0.8, edgecolor="white")
axes[1].set_title("Distribusi Frekuensi (Log Scale)", fontweight="bold")
axes[1].set_xlabel("log(1 + n_invoices)")
axes[1].set_ylabel("Jumlah Pelanggan")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_customer_frequency_dist.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["customer_frequency"] = {
    "n_customers": int(len(cust_freq)),
    "mean_freq": round(float(cust_freq["n_invoices"].mean()), 2),
    "median_freq": float(cust_freq["n_invoices"].median()),
    "p75_freq": float(cust_freq["n_invoices"].quantile(0.75)),
    "p90_freq": float(cust_freq["n_invoices"].quantile(0.90)),
    "p99_freq": float(cust_freq["n_invoices"].quantile(0.99)),
    "max_freq": int(cust_freq["n_invoices"].max()),
    "pct_single_tx": round((cust_freq["n_invoices"] == 1).mean() * 100, 2),
    "pct_ge5_tx": round((cust_freq["n_invoices"] >= 5).mean() * 100, 2),
}
print(f"  Median frekuensi: {eda_stats['customer_frequency']['median_freq']}")

# ============================================================
# 3. DISTRIBUSI RECENCY
# ============================================================
print("Plot 3: Distribusi recency...")
# Recency dari perspektif akhir dataset
max_date = df_m["InvoiceDate"].max()
last_purchase = df_m.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_purchase.columns = ["CustomerID", "last_purchase"]
last_purchase["recency_days"] = (max_date - last_purchase["last_purchase"]).dt.days

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(last_purchase["recency_days"], bins=50, color=COLORS[4], alpha=0.8, edgecolor="white")
axes[0].set_title(f"Distribusi Recency\n(relatif terhadap {max_date.date()})", fontweight="bold")
axes[0].set_xlabel("Hari Sejak Pembelian Terakhir")
axes[0].set_ylabel("Jumlah Pelanggan")
axes[0].axvline(last_purchase["recency_days"].median(), color="red", linestyle="--",
                label=f"Median={last_purchase['recency_days'].median():.0f} hari")
axes[0].axvline(90, color="orange", linestyle=":", label="90-hari threshold")
axes[0].legend()

# Persentil kumulatif
sorted_rec = np.sort(last_purchase["recency_days"])
pct = np.arange(1, len(sorted_rec)+1) / len(sorted_rec) * 100
axes[1].plot(sorted_rec, pct, color=COLORS[0], linewidth=2)
axes[1].axvline(90, color="orange", linestyle=":", label="90 hari")
axes[1].axvline(70, color="green", linestyle=":", label="70 hari")
axes[1].set_title("Distribusi Kumulatif Recency", fontweight="bold")
axes[1].set_xlabel("Hari Sejak Pembelian Terakhir")
axes[1].set_ylabel("% Pelanggan")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_recency_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["recency"] = {
    "recency_reference_date": str(max_date.date()),
    "mean_recency": round(float(last_purchase["recency_days"].mean()), 2),
    "median_recency": float(last_purchase["recency_days"].median()),
    "pct_active_within_30d": round((last_purchase["recency_days"] <= 30).mean() * 100, 2),
    "pct_active_within_70d": round((last_purchase["recency_days"] <= 70).mean() * 100, 2),
    "pct_active_within_90d": round((last_purchase["recency_days"] <= 90).mean() * 100, 2),
    "pct_inactive_gt90d": round((last_purchase["recency_days"] > 90).mean() * 100, 2),
}
print(f"  Median recency: {eda_stats['recency']['median_recency']} hari")
print(f"  Aktif <= 90 hari: {eda_stats['recency']['pct_active_within_90d']:.1f}%")

# ============================================================
# 4. DISTRIBUSI MONETARY (REVENUE PER PELANGGAN)
# ============================================================
print("Plot 4: Distribusi monetary...")
cust_revenue = df_m.groupby("CustomerID")["LineValue"].sum().reset_index()
cust_revenue.columns = ["CustomerID", "total_revenue"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Raw distribution (capped)
cap_val = cust_revenue["total_revenue"].quantile(0.95)
axes[0].hist(cust_revenue["total_revenue"].clip(upper=cap_val), bins=50, color=COLORS[0], alpha=0.8, edgecolor="white")
axes[0].set_title("Revenue per Pelanggan\n(dikap P95)", fontweight="bold")
axes[0].set_xlabel("Total Revenue (GBP)")
axes[0].set_ylabel("Jumlah Pelanggan")

# Log distribution
axes[1].hist(np.log1p(cust_revenue["total_revenue"]), bins=50, color=COLORS[1], alpha=0.8, edgecolor="white")
axes[1].set_title("Revenue per Pelanggan (Log Scale)", fontweight="bold")
axes[1].set_xlabel("log(1 + Revenue)")
axes[1].set_ylabel("Jumlah Pelanggan")

# Lorenz curve (revenue concentration)
sorted_rev = np.sort(cust_revenue["total_revenue"])
cumrev = np.cumsum(sorted_rev) / sorted_rev.sum()
cumpop = np.arange(1, len(sorted_rev)+1) / len(sorted_rev)
axes[2].plot(cumpop * 100, cumrev * 100, color=COLORS[2], linewidth=2, label="Lorenz Curve")
axes[2].plot([0, 100], [0, 100], "k--", linewidth=1, label="Perfect Equality")
axes[2].fill_between(cumpop * 100, cumpop * 100, cumrev * 100, alpha=0.1, color=COLORS[2])
axes[2].set_title("Kurva Lorenz — Konsentrasi Revenue", fontweight="bold")
axes[2].set_xlabel("% Pelanggan (kumulatif)")
axes[2].set_ylabel("% Revenue (kumulatif)")
axes[2].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_monetary_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Hitung Gini coefficient
n = len(sorted_rev)
gini = (2 * np.sum(np.arange(1, n+1) * sorted_rev) / (n * sorted_rev.sum())) - (n + 1) / n
# Revenue share top 20%
top20_threshold = np.percentile(sorted_rev, 80)
top20_rev_share = sorted_rev[sorted_rev >= top20_threshold].sum() / sorted_rev.sum()

eda_stats["monetary"] = {
    "mean_revenue_per_customer": round(float(cust_revenue["total_revenue"].mean()), 2),
    "median_revenue_per_customer": round(float(cust_revenue["total_revenue"].median()), 2),
    "p75_revenue": round(float(cust_revenue["total_revenue"].quantile(0.75)), 2),
    "p90_revenue": round(float(cust_revenue["total_revenue"].quantile(0.90)), 2),
    "p99_revenue": round(float(cust_revenue["total_revenue"].quantile(0.99)), 2),
    "max_revenue": round(float(cust_revenue["total_revenue"].max()), 2),
    "gini_coefficient": round(float(gini), 4),
    "top20pct_customers_revenue_share": round(float(top20_rev_share) * 100, 2),
}
print(f"  Gini coefficient revenue: {eda_stats['monetary']['gini_coefficient']:.3f}")
print(f"  Top 20% pelanggan = {eda_stats['monetary']['top20pct_customers_revenue_share']:.1f}% revenue")

# ============================================================
# 5. RATA-RATA ORDER VALUE (AOV)
# ============================================================
print("Plot 5: Average Order Value...")
invoice_value = df_m.groupby("InvoiceNo")["LineValue"].sum().reset_index()
invoice_value.columns = ["InvoiceNo", "order_value"]
invoice_value = invoice_value[invoice_value["order_value"] > 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cap_aov = invoice_value["order_value"].quantile(0.97)
axes[0].hist(invoice_value["order_value"].clip(upper=cap_aov), bins=60, color=COLORS[3], alpha=0.8, edgecolor="white")
axes[0].set_title("Distribusi Nilai per Order\n(dikap P97)", fontweight="bold")
axes[0].set_xlabel("Order Value (GBP)")
axes[0].set_ylabel("Jumlah Order")
axes[0].axvline(invoice_value["order_value"].median(), color="red", linestyle="--",
                label=f"Median=£{invoice_value['order_value'].median():.0f}")
axes[0].legend()

axes[1].hist(np.log1p(invoice_value["order_value"]), bins=50, color=COLORS[4], alpha=0.8, edgecolor="white")
axes[1].set_title("Distribusi Nilai per Order (Log)", fontweight="bold")
axes[1].set_xlabel("log(1 + Order Value)")
axes[1].set_ylabel("Jumlah Order")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "05_aov_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["aov"] = {
    "mean_aov": round(float(invoice_value["order_value"].mean()), 2),
    "median_aov": round(float(invoice_value["order_value"].median()), 2),
    "p90_aov": round(float(invoice_value["order_value"].quantile(0.90)), 2),
    "p99_aov": round(float(invoice_value["order_value"].quantile(0.99)), 2),
    "max_aov": round(float(invoice_value["order_value"].max()), 2),
}
print(f"  Median AOV: GBP {eda_stats['aov']['median_aov']:.2f}")

# ============================================================
# 6. DISTRIBUSI NEGARA
# ============================================================
print("Plot 6: Distribusi negara...")
country_rev = df_m.groupby("Country")["LineValue"].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(country_rev.index[::-1], country_rev.values[::-1] / 1000, color=COLORS[0], alpha=0.8)
ax.set_title("Top 10 Negara berdasarkan Revenue", fontweight="bold", fontsize=13)
ax.set_xlabel("Revenue (GBP Thousands)")
for bar, val in zip(bars, country_rev.values[::-1] / 1000):
    ax.text(val + 5, bar.get_y() + bar.get_height() / 2, f"£{val:.0f}k", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "06_country_revenue.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["geography"] = {
    "top_countries_revenue": {k: round(float(v), 2) for k, v in country_rev.items()},
    "uk_revenue_share": round(float(country_rev.get("United Kingdom", 0) / country_rev.sum() * 100), 2),
}

# ============================================================
# 7. CANCELLATION ANALYSIS
# ============================================================
print("Plot 7: Analisis cancellation...")
df_analysis_dated = df_a.copy()
df_analysis_dated["InvoiceDate"] = pd.to_datetime(df_analysis_dated["InvoiceDate"])
df_analysis_dated["YM"] = df_analysis_dated["InvoiceDate"].dt.to_period("M")

cancel_by_month = df_analysis_dated.groupby(["YM", "IsCancelled"])["InvoiceNo"].nunique().unstack(fill_value=0)
cancel_by_month.columns = ["Normal", "Cancelled"]
cancel_by_month["CancelRate"] = cancel_by_month["Cancelled"] / (cancel_by_month["Normal"] + cancel_by_month["Cancelled"]) * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
cancel_by_month_str = cancel_by_month.copy()
cancel_by_month_str.index = cancel_by_month_str.index.astype(str)

axes[0].bar(cancel_by_month_str.index, cancel_by_month_str["Normal"], label="Normal", color=COLORS[0], alpha=0.8)
axes[0].bar(cancel_by_month_str.index, cancel_by_month_str["Cancelled"], bottom=cancel_by_month_str["Normal"],
            label="Cancelled", color=COLORS[5], alpha=0.8)
axes[0].set_title("Invoice Normal vs Cancelled per Bulan", fontweight="bold")
axes[0].set_ylabel("Jumlah Invoice")
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend()

axes[1].plot(cancel_by_month_str.index, cancel_by_month_str["CancelRate"], marker="o", color=COLORS[5], linewidth=2)
axes[1].set_title("Cancellation Rate (%) per Bulan", fontweight="bold")
axes[1].set_ylabel("Cancellation Rate (%)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].axhline(cancel_by_month_str["CancelRate"].mean(), color="red", linestyle="--",
                label=f"Rata-rata: {cancel_by_month_str['CancelRate'].mean():.1f}%")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_cancellation_analysis.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["cancellations"] = {
    "mean_monthly_cancel_rate_pct": round(float(cancel_by_month["CancelRate"].mean()), 2),
    "max_monthly_cancel_rate_pct": round(float(cancel_by_month["CancelRate"].max()), 2),
    "peak_cancel_month": str(cancel_by_month["CancelRate"].idxmax()),
}
print(f"  Rata-rata cancel rate bulanan: {eda_stats['cancellations']['mean_monthly_cancel_rate_pct']:.1f}%")

# ============================================================
# 8. REPEAT PURCHASE BEHAVIOR
# ============================================================
print("Plot 8: Repeat purchase behavior...")
# Hitung inter-purchase time (IPT) untuk pelanggan >= 2 transaksi
cust_dates = df_m.groupby("CustomerID")["InvoiceDate"].agg(list).reset_index()
ipt_list = []
for _, row in cust_dates.iterrows():
    dates = sorted(pd.to_datetime(row["InvoiceDate"]))
    if len(dates) >= 2:
        for i in range(1, len(dates)):
            diff = (dates[i].date() - dates[i-1].date()).days
            ipt_list.append(diff)

ipt_series = pd.Series(ipt_list)
ipt_positive = ipt_series[ipt_series > 0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(ipt_positive.clip(upper=200), bins=60, color=COLORS[1], alpha=0.8, edgecolor="white")
axes[0].set_title("Inter-Purchase Time Distribution\n(dikap 200 hari)", fontweight="bold")
axes[0].set_xlabel("Hari antara Pembelian Berurutan")
axes[0].set_ylabel("Frekuensi")
axes[0].axvline(ipt_positive.median(), color="red", linestyle="--",
                label=f"Median={ipt_positive.median():.0f} hari")
axes[0].axvline(90, color="orange", linestyle=":", label="90-hari threshold")
axes[0].legend()

# Proporsi pelanggan berdasarkan jumlah pembelian
bucket_labels = ["1 tx", "2-4 tx", "5-9 tx", "10-19 tx", "20+ tx"]
buckets = [
    (cust_freq["n_invoices"] == 1).sum(),
    ((cust_freq["n_invoices"] >= 2) & (cust_freq["n_invoices"] <= 4)).sum(),
    ((cust_freq["n_invoices"] >= 5) & (cust_freq["n_invoices"] <= 9)).sum(),
    ((cust_freq["n_invoices"] >= 10) & (cust_freq["n_invoices"] <= 19)).sum(),
    (cust_freq["n_invoices"] >= 20).sum(),
]
axes[1].pie(buckets, labels=bucket_labels, autopct="%1.1f%%", colors=sns.color_palette("muted", 5))
axes[1].set_title("Proporsi Pelanggan per Segmen Frekuensi", fontweight="bold")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_repeat_purchase.png", dpi=120, bbox_inches="tight")
plt.close()

eda_stats["inter_purchase_time"] = {
    "n_ipt_observations": int(len(ipt_positive)),
    "median_ipt_days": round(float(ipt_positive.median()), 2),
    "mean_ipt_days": round(float(ipt_positive.mean()), 2),
    "pct_ipt_lt30d": round((ipt_positive <= 30).mean() * 100, 2),
    "pct_ipt_lt90d": round((ipt_positive <= 90).mean() * 100, 2),
    "pct_ipt_gt90d": round((ipt_positive > 90).mean() * 100, 2),
}
print(f"  Median IPT: {eda_stats['inter_purchase_time']['median_ipt_days']:.0f} hari")
print(f"  % gap > 90 hari: {eda_stats['inter_purchase_time']['pct_ipt_gt90d']:.1f}%")

# ============================================================
# 9. COHORT RETENTION (SIMPLIFIED)
# ============================================================
print("Plot 9: Cohort retention...")
df_m["CohortMonth"] = df_m.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M")
df_m["InvoicePeriod"] = df_m["InvoiceDate"].dt.to_period("M")

cohort_data = df_m.groupby(["CohortMonth", "InvoicePeriod"])["CustomerID"].nunique().reset_index()
cohort_data["CohortIndex"] = (
    cohort_data["InvoicePeriod"].dt.to_timestamp() - cohort_data["CohortMonth"].dt.to_timestamp()
).dt.days // 30

cohort_pivot = cohort_data.pivot_table(index="CohortMonth", columns="CohortIndex", values="CustomerID", aggfunc="sum")
cohort_sizes = cohort_pivot[0]
retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

# Hanya tampilkan cohort yang punya cukup follow-up (hingga index 6)
ret_display = retention_matrix.iloc[:, :7].copy()
ret_display.index = ret_display.index.astype(str)

fig, ax = plt.subplots(figsize=(13, 7))
mask = ret_display.isnull()
sns.heatmap(
    ret_display, annot=True, fmt=".0f", mask=mask,
    cmap="YlOrRd_r", linewidths=0.5, ax=ax,
    cbar_kws={"label": "% Pelanggan Aktif"},
    vmin=0, vmax=100,
)
ax.set_title("Cohort Retention Rate (%)\n(Bulan 0 = bulan pertama pembelian)", fontweight="bold", fontsize=12)
ax.set_xlabel("Bulan sejak Pembelian Pertama")
ax.set_ylabel("Cohort (Bulan Pertama Beli)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_cohort_retention.png", dpi=120, bbox_inches="tight")
plt.close()

# Rata-rata retention di bulan 1, 3, 6
try:
    avg_m1 = round(float(retention_matrix[1].dropna().mean()), 2)
    avg_m3 = round(float(retention_matrix[3].dropna().mean()), 2) if 3 in retention_matrix.columns else None
    avg_m6 = round(float(retention_matrix[6].dropna().mean()), 2) if 6 in retention_matrix.columns else None
except:
    avg_m1, avg_m3, avg_m6 = None, None, None

eda_stats["cohort_retention"] = {
    "avg_month1_retention_pct": avg_m1,
    "avg_month3_retention_pct": avg_m3,
    "avg_month6_retention_pct": avg_m6,
}
print(f"  Avg retention bulan 1: {avg_m1}%")
print(f"  Avg retention bulan 3: {avg_m3}%")

# ============================================================
# 10. TEMPORAL DESIGN VALIDATION
# ============================================================
print("\nValidasi desain temporal...")

# Hitung distribusi pelanggan per possible cutoff
cutoffs = {
    "2011-08-31": pd.Timestamp("2011-08-31"),
    "2011-09-30": pd.Timestamp("2011-09-30"),
    "2011-10-31": pd.Timestamp("2011-10-31"),
}
horizons = {90: 90, 70: 70, 60: 60}

temporal_analysis = {}
for cutoff_name, cutoff_date in cutoffs.items():
    for horizon_days in [60, 70, 90]:
        horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)

        # Pelanggan eligible: punya >= 1 tx dalam 90 hari sebelum cutoff
        obs_start = cutoff_date - pd.Timedelta(days=90)
        eligible_mask = (
            (df_m["InvoiceDate"] >= obs_start)
            & (df_m["InvoiceDate"] <= cutoff_date)
        )
        eligible_customers = df_m[eligible_mask]["CustomerID"].unique()
        n_eligible = len(eligible_customers)

        # Dari eligible, siapa yang beli di horizon?
        horizon_mask = (
            (df_m["InvoiceDate"] > cutoff_date)
            & (df_m["InvoiceDate"] <= horizon_end)
        )
        active_in_horizon = df_m[horizon_mask]["CustomerID"].unique()
        active_eligible = set(eligible_customers) & set(active_in_horizon)

        n_active = len(active_eligible)
        n_inactive = n_eligible - n_active
        pct_inactive = n_inactive / n_eligible * 100 if n_eligible > 0 else 0

        key = f"cutoff={cutoff_name}_horizon={horizon_days}d"
        temporal_analysis[key] = {
            "cutoff": cutoff_name,
            "horizon_days": horizon_days,
            "horizon_end": str(horizon_end.date()),
            "n_eligible_customers": n_eligible,
            "n_active_in_horizon": n_active,
            "n_inactive": n_inactive,
            "pct_inactive": round(pct_inactive, 2),
            "data_covers_full_horizon": str(horizon_end.date()) <= "2011-12-09",
        }

eda_stats["temporal_design"] = temporal_analysis

# Visualisasi temporal scenarios
scenarios = pd.DataFrame(list(temporal_analysis.values()))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: n_eligible vs scenario
scenarios["label"] = "c=" + scenarios["cutoff"].str[-5:] + "\nh=" + scenarios["horizon_days"].astype(str) + "d"
axes[0].bar(range(len(scenarios)), scenarios["n_eligible_customers"], color=COLORS[0], alpha=0.8)
axes[0].set_xticks(range(len(scenarios)))
axes[0].set_xticklabels(scenarios["label"], fontsize=8, rotation=30)
axes[0].set_title("Jumlah Pelanggan Eligible\nper Skenario Temporal", fontweight="bold")
axes[0].set_ylabel("Jumlah Pelanggan")

# Plot 2: % inactive vs scenario
colors_bar = ["green" if s else "red" for s in scenarios["data_covers_full_horizon"]]
bars = axes[1].bar(range(len(scenarios)), scenarios["pct_inactive"], color=colors_bar, alpha=0.8)
axes[1].set_xticks(range(len(scenarios)))
axes[1].set_xticklabels(scenarios["label"], fontsize=8, rotation=30)
axes[1].set_title("% Pelanggan Inactive per Skenario\n(Hijau = horizon tersedia penuh)", fontweight="bold")
axes[1].set_ylabel("% Inactive")
axes[1].axhline(50, color="black", linestyle="--", linewidth=1, label="50%")
axes[1].legend()
for bar, val in zip(bars, scenarios["pct_inactive"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_temporal_design_scenarios.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================
# PRINT TEMPORAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN TEMPORAL DESIGN VALIDATION")
print("=" * 65)
print(f"{'Skenario':<45} {'Eligible':>9} {'Inactive':>9} {'%Inact':>7} {'Full?':>6}")
print("-" * 65)
for k, v in temporal_analysis.items():
    full = "YES" if v["data_covers_full_horizon"] else "NO"
    print(f"  cutoff={v['cutoff']}, horizon={v['horizon_days']}d: "
          f"{v['n_eligible_customers']:>7,}  {v['n_inactive']:>7,}  {v['pct_inactive']:>6.1f}%  {full}")

# ============================================================
# SIMPAN EDA STATS
# ============================================================
with open(REPORTS_DIR / "eda_stats.json", "w", encoding="utf-8") as f:
    json.dump(eda_stats, f, indent=2, default=str)

print("\n" + "=" * 65)
print("EDA SELESAI")
print("=" * 65)
print("\nPlot tersimpan:")
for p in sorted(PLOTS_DIR.glob("*.png")):
    print(f"  {p.name}")
print(f"\nStats tersimpan: {REPORTS_DIR / 'eda_stats.json'}")
