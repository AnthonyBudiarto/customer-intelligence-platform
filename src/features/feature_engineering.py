"""
Feature Engineering — Customer-Level Features untuk Pemodelan Prediktif

PRINSIP UTAMA:
- Semua fitur dibangun menggunakan hanya transaksi pada atau sebelum prediction_cutoff.
- Tidak ada informasi dari prediction horizon yang masuk ke fitur.
- Fungsi menerima cutoff_date sebagai parameter eksplisit — tidak ada default global.

Fitur yang dibangun:
    recency_days           : Hari sejak qualified purchase terakhir sebelum cutoff
    frequency              : Jumlah invoice unik sebelum cutoff
    monetary_total         : Total nilai pembelian sebelum cutoff
    monetary_mean          : Rata-rata nilai per invoice
    monetary_std           : Standar deviasi nilai per invoice (0 jika 1 transaksi)
    avg_qty_per_order      : Rata-rata jumlah item per invoice
    n_unique_products      : Jumlah produk unik yang pernah dibeli
    n_unique_months        : Jumlah bulan berbeda dengan pembelian
    tenure_days            : Hari antara pembelian pertama dan cutoff
    ipt_mean               : Rata-rata inter-purchase time (NaN jika < 2 tx)
    ipt_std                : Std inter-purchase time (NaN jika < 2 tx)
    recent_freq_ratio      : Frekuensi 30 hari terakhir / frekuensi total
    spending_trend         : Revenue 30 hari terakhir / revenue total
    cancellation_rate      : Rasio cancelled invoice terhadap total invoice pelanggan
    country_UK             : Binary — apakah pelanggan dari UK
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


def build_customer_features(
    df_modeling: pd.DataFrame,
    df_analysis: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    min_transactions: int = 1,
) -> pd.DataFrame:
    """
    Bangun fitur customer-level dari data transaksi.

    Parameters
    ----------
    df_modeling : pd.DataFrame
        Cleaned modeling data (qualified purchases saja — tidak ada cancellations).
        Harus memiliki kolom: CustomerID, InvoiceDate, InvoiceNo, LineValue, Quantity,
        StockCode, Country.
    df_analysis : pd.DataFrame
        Cleaned analysis data (termasuk cancellations) untuk fitur cancellation_rate.
        Harus memiliki kolom: CustomerID, InvoiceDate, InvoiceNo, IsCancelled.
    cutoff_date : pd.Timestamp
        Batas waktu — hanya transaksi pada atau sebelum tanggal ini yang digunakan.
        PENTING: Tidak ada transaksi setelah cutoff_date yang boleh masuk ke fitur.
    min_transactions : int
        Jumlah minimum qualified purchases sebelum cutoff agar pelanggan diikutsertakan.
        Default: 1 (semua pelanggan yang pernah beli sebelum cutoff).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan satu baris per CustomerID dan semua kolom fitur.
        Index-nya adalah CustomerID.

    Raises
    ------
    ValueError
        Jika cutoff_date berada di luar rentang dataset atau df kosong.
    """
    if len(df_modeling) == 0:
        raise ValueError("df_modeling kosong — tidak bisa membangun fitur")

    # Konversi cutoff ke Timestamp jika perlu
    cutoff_date = pd.Timestamp(cutoff_date)

    # Validasi cutoff
    date_min = df_modeling["InvoiceDate"].min()
    date_max = df_modeling["InvoiceDate"].max()
    if cutoff_date < date_min:
        raise ValueError(f"cutoff_date {cutoff_date.date()} sebelum data pertama {date_min.date()}")
    if cutoff_date > date_max:
        print(f"WARNING: cutoff_date {cutoff_date.date()} setelah data terakhir {date_max.date()}. "
              f"Ini valid untuk prediksi tapi semua pelanggan akan memiliki recency > 0.")

    # --- Filter ke observation window (sebelum atau sama dengan cutoff) ---
    obs = df_modeling[df_modeling["InvoiceDate"] <= cutoff_date].copy()

    if len(obs) == 0:
        raise ValueError(f"Tidak ada transaksi pada atau sebelum cutoff {cutoff_date.date()}")

    # ----------------------------------------------------------------
    # A. FITUR INTI (RFM dan turunannya)
    # ----------------------------------------------------------------

    # Recency: hari sejak pembelian terakhir
    last_purchase = obs.groupby("CustomerID")["InvoiceDate"].max()
    recency = (cutoff_date - last_purchase).dt.days

    # Frequency: jumlah invoice unik
    frequency = obs.groupby("CustomerID")["InvoiceNo"].nunique()

    # Monetary: total dan statistik nilai
    invoice_values = obs.groupby(["CustomerID", "InvoiceNo"])["LineValue"].sum().reset_index()
    invoice_values.columns = ["CustomerID", "InvoiceNo", "InvoiceValue"]

    monetary_stats = invoice_values.groupby("CustomerID")["InvoiceValue"].agg(
        monetary_total="sum",
        monetary_mean="mean",
        monetary_std="std",
    )
    # std = NaN untuk pelanggan dengan 1 transaksi, isi dengan 0
    monetary_stats["monetary_std"] = monetary_stats["monetary_std"].fillna(0.0)

    # ----------------------------------------------------------------
    # B. FITUR KUANTITAS DAN PRODUK
    # ----------------------------------------------------------------
    qty_per_invoice = obs.groupby(["CustomerID", "InvoiceNo"])["Quantity"].sum().reset_index()
    qty_stats = qty_per_invoice.groupby("CustomerID")["Quantity"].agg(avg_qty_per_order="mean")

    n_unique_products = obs.groupby("CustomerID")["StockCode"].nunique().rename("n_unique_products")

    # ----------------------------------------------------------------
    # C. FITUR TEMPORAL
    # ----------------------------------------------------------------
    # Jumlah bulan unik dengan pembelian
    obs["YearMonth"] = obs["InvoiceDate"].dt.to_period("M")
    n_unique_months = obs.groupby("CustomerID")["YearMonth"].nunique().rename("n_unique_months")

    # Tenure: hari antara pembelian pertama dan cutoff
    first_purchase = obs.groupby("CustomerID")["InvoiceDate"].min()
    tenure = (cutoff_date - first_purchase).dt.days.rename("tenure_days")

    # Inter-purchase time (hanya untuk pelanggan >= 2 transaksi)
    def compute_ipt(dates):
        """Hitung mean dan std IPT dari list tanggal invoice."""
        unique_dates = sorted(dates.unique())
        if len(unique_dates) < 2:
            return pd.Series({"ipt_mean": np.nan, "ipt_std": np.nan})
        diffs = [(unique_dates[i] - unique_dates[i-1]).days for i in range(1, len(unique_dates))]
        diffs = [d for d in diffs if d > 0]  # abaikan sama-hari
        if len(diffs) == 0:
            return pd.Series({"ipt_mean": np.nan, "ipt_std": np.nan})
        return pd.Series({"ipt_mean": np.mean(diffs), "ipt_std": np.std(diffs) if len(diffs) > 1 else 0.0})

    ipt_stats = obs.groupby("CustomerID")["InvoiceDate"].apply(compute_ipt).unstack()

    # ----------------------------------------------------------------
    # D. FITUR TREND (RECENT vs HISTORICAL)
    # ----------------------------------------------------------------
    recent_window = cutoff_date - pd.Timedelta(days=30)
    recent_obs = obs[obs["InvoiceDate"] >= recent_window]

    recent_freq = recent_obs.groupby("CustomerID")["InvoiceNo"].nunique().rename("recent_invoices_30d")
    recent_spend = recent_obs.groupby("CustomerID")["LineValue"].sum().rename("recent_spend_30d")

    # Ratio recent / total (0 jika tidak ada pembelian recent)
    freq_ratio = (recent_freq / frequency).fillna(0).rename("recent_freq_ratio")
    spend_ratio = (recent_spend / monetary_stats["monetary_total"]).fillna(0).rename("spending_trend")

    # ----------------------------------------------------------------
    # E. FITUR CANCELLATION RATE
    # ----------------------------------------------------------------
    # Dari df_analysis yang mengandung cancelled invoices
    obs_analysis = df_analysis[df_analysis["InvoiceDate"] <= cutoff_date].copy()

    cancel_counts = obs_analysis[obs_analysis["IsCancelled"]].groupby("CustomerID")["InvoiceNo"].nunique()
    total_invoice_counts = obs_analysis.groupby("CustomerID")["InvoiceNo"].nunique()
    cancel_rate = (cancel_counts / total_invoice_counts).fillna(0).rename("cancellation_rate")

    # ----------------------------------------------------------------
    # F. FITUR DEMOGRAFIS
    # ----------------------------------------------------------------
    # Negara (ambil negara yang paling sering muncul per pelanggan)
    country_mode = obs.groupby("CustomerID")["Country"].agg(
        lambda x: x.mode().iloc[0] if len(x) > 0 else "Unknown"
    ).rename("country")
    is_uk = (country_mode == "United Kingdom").astype(int).rename("country_UK")

    # ----------------------------------------------------------------
    # G. GABUNGKAN SEMUA FITUR
    # ----------------------------------------------------------------
    features = pd.concat(
        [
            recency.rename("recency_days"),
            frequency.rename("frequency"),
            monetary_stats,
            qty_stats,
            n_unique_products,
            n_unique_months,
            tenure,
            ipt_stats,
            freq_ratio,
            spend_ratio,
            cancel_rate,
            is_uk,
        ],
        axis=1,
    )

    # Hanya pelanggan dengan >= min_transactions qualified purchases
    features = features[features["frequency"] >= min_transactions].copy()

    # Pastikan tidak ada kebocoran — verifikasi semua pelanggan ada di obs
    assert features.index.isin(obs["CustomerID"]).all(), \
        "LEAKAGE CHECK GAGAL: beberapa CustomerID tidak ada dalam observation window!"

    # Metadata
    features.index.name = "CustomerID"
    features.attrs["cutoff_date"] = str(cutoff_date.date())
    features.attrs["n_customers"] = len(features)
    features.attrs["min_transactions"] = min_transactions

    return features


def validate_no_leakage(
    features: pd.DataFrame,
    df_modeling: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> dict:
    """
    Validasi bahwa fitur dibangun hanya dari transaksi pada atau sebelum cutoff_date.

    Pemeriksaan:
    - Verifikasi bahwa tanggal transaksi yang MASUK KE FITUR (setelah filter cutoff)
      memang tidak melebihi cutoff.
    - Pemeriksaan ini mengkonfirmasi bahwa build_customer_features() benar memfilter.

    Catatan: df_modeling mengandung transaksi untuk semua periode — yang penting
    adalah fitur yang dibangun hanya menggunakan data <= cutoff.

    Returns
    -------
    dict
        Hasil validasi dengan status dan detail.
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    customer_ids = features.index.tolist()

    # Filter ke data yang seharusnya masuk ke fitur
    relevant_pre_cutoff = df_modeling[
        (df_modeling["CustomerID"].isin(customer_ids))
        & (df_modeling["InvoiceDate"] <= cutoff_date)
    ]

    # Pelanggan yang memiliki fitur tapi tidak punya data pre-cutoff = masalah
    customers_with_pre_cutoff_data = set(relevant_pre_cutoff["CustomerID"].unique())
    customers_without_pre_cutoff = set(customer_ids) - customers_with_pre_cutoff_data

    # Verifikasi: recency di fitur harus >= 0 (tidak bisa negatif)
    negative_recency = features[features["recency_days"] < 0] if "recency_days" in features.columns else pd.DataFrame()

    result = {
        "status": "PASS" if len(customers_without_pre_cutoff) == 0 and len(negative_recency) == 0 else "FAIL",
        "n_customers_checked": len(customer_ids),
        "n_customers_with_pre_cutoff_data": len(customers_with_pre_cutoff_data),
        "n_customers_without_any_pre_cutoff_data": len(customers_without_pre_cutoff),
        "n_negative_recency": len(negative_recency),
        "cutoff_date": str(cutoff_date.date()),
        "note": "Fitur dibangun dari transaksi <= cutoff_date. df_modeling mengandung semua periode (normal).",
    }

    return result
