"""
Modul audit data: investigasi kualitas dataset raw tanpa memodifikasinya.

Setiap fungsi menghasilkan temuan yang dicatat ke audit report.
Tidak ada data yang dihapus atau diubah di sini.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from src.utils.logger import get_logger

logger = get_logger("data.auditor")


def audit_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Dimensi, tipe data, dan penggunaan memori."""
    logger.info("Menjalankan audit: informasi dasar dataset")

    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "columns": list(df.columns),
    }

    logger.info(f"  Baris: {report['n_rows']:,}")
    logger.info(f"  Kolom: {report['n_columns']}")
    logger.info(f"  Memori: {report['memory_mb']:.1f} MB")
    return report


def audit_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Hitung nilai hilang per kolom."""
    logger.info("Menjalankan audit: nilai yang hilang (missing values)")

    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    report = {
        "missing_count": missing_count.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "total_missing_cells": int(missing_count.sum()),
        "rows_with_null_customer_id": int(df["CustomerID"].isnull().sum()),
        "pct_null_customer_id": round(df["CustomerID"].isnull().mean() * 100, 2),
    }

    for col, count in missing_count.items():
        if count > 0:
            logger.warning(f"  {col}: {count:,} missing ({missing_pct[col]:.1f}%)")

    logger.info(
        f"  CustomerID hilang: {report['rows_with_null_customer_id']:,} baris "
        f"({report['pct_null_customer_id']:.1f}%)"
    )
    return report


def audit_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Deteksi baris duplikat identik."""
    logger.info("Menjalankan audit: duplikat")

    n_duplicates = df.duplicated().sum()
    report = {
        "n_duplicate_rows": int(n_duplicates),
        "pct_duplicate_rows": round(n_duplicates / len(df) * 100, 2),
    }

    logger.info(
        f"  Duplikat: {n_duplicates:,} baris ({report['pct_duplicate_rows']:.1f}%)"
    )
    return report


def audit_date_range(df: pd.DataFrame) -> Dict[str, Any]:
    """Periksa rentang tanggal dan missing timestamps."""
    logger.info("Menjalankan audit: rentang tanggal")

    dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    n_invalid_dates = dates.isnull().sum()

    report = {
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "date_range_days": (dates.max() - dates.min()).days,
        "n_invalid_dates": int(n_invalid_dates),
        "n_unique_dates": int(dates.dt.date.nunique()),
        "n_unique_months": int(dates.dt.to_period("M").nunique()),
    }

    logger.info(f"  Tanggal awal: {report['date_min']}")
    logger.info(f"  Tanggal akhir: {report['date_max']}")
    logger.info(f"  Rentang: {report['date_range_days']} hari")
    logger.info(f"  Bulan unik: {report['n_unique_months']}")
    if n_invalid_dates > 0:
        logger.warning(f"  Tanggal tidak valid: {n_invalid_dates:,}")
    return report


def audit_cancelled_invoices(df: pd.DataFrame) -> Dict[str, Any]:
    """Identifikasi invoice yang dibatalkan (prefix 'C')."""
    logger.info("Menjalankan audit: invoice yang dibatalkan")

    is_cancelled = df["InvoiceNo"].str.startswith("C", na=False)
    cancelled_df = df[is_cancelled]

    report = {
        "n_cancelled_rows": int(is_cancelled.sum()),
        "pct_cancelled_rows": round(is_cancelled.mean() * 100, 2),
        "n_cancelled_invoices": int(cancelled_df["InvoiceNo"].nunique()),
        "n_customers_with_cancellations": int(
            cancelled_df["CustomerID"].dropna().nunique()
        ),
        "total_cancelled_quantity": int(cancelled_df["Quantity"].sum()),
        "total_cancelled_value": round(
            (cancelled_df["Quantity"] * cancelled_df["UnitPrice"]).sum(), 2
        ),
    }

    logger.info(f"  Baris cancelled: {report['n_cancelled_rows']:,} ({report['pct_cancelled_rows']:.1f}%)")
    logger.info(f"  Invoice cancelled unik: {report['n_cancelled_invoices']:,}")
    logger.info(f"  Pelanggan dengan cancellations: {report['n_customers_with_cancellations']:,}")
    return report


def audit_quantity_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Periksa quantity negatif, nol, dan ekstrem."""
    logger.info("Menjalankan audit: masalah quantity")

    # Hanya untuk non-cancelled invoices
    non_cancelled = df[~df["InvoiceNo"].str.startswith("C", na=False)]

    negative_qty = non_cancelled[non_cancelled["Quantity"] < 0]
    zero_qty = non_cancelled[non_cancelled["Quantity"] == 0]

    report = {
        "n_negative_quantity_non_cancelled": int(len(negative_qty)),
        "n_zero_quantity": int(len(zero_qty)),
        "quantity_min": float(df["Quantity"].min()),
        "quantity_max": float(df["Quantity"].max()),
        "quantity_p99": float(df["Quantity"].quantile(0.99)),
        "quantity_p999": float(df["Quantity"].quantile(0.999)),
    }

    logger.info(f"  Quantity min: {report['quantity_min']:,}")
    logger.info(f"  Quantity max: {report['quantity_max']:,}")
    logger.info(f"  Quantity negatif (non-cancelled): {report['n_negative_quantity_non_cancelled']:,}")
    logger.info(f"  Quantity = 0: {report['n_zero_quantity']:,}")
    return report


def audit_price_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Periksa harga nol, negatif, dan ekstrem."""
    logger.info("Menjalankan audit: masalah unit price")

    report = {
        "n_zero_price": int((df["UnitPrice"] == 0).sum()),
        "n_negative_price": int((df["UnitPrice"] < 0).sum()),
        "price_min": float(df["UnitPrice"].min()),
        "price_max": float(df["UnitPrice"].max()),
        "price_p99": float(df["UnitPrice"].quantile(0.99)),
        "price_p999": float(df["UnitPrice"].quantile(0.999)),
        "pct_zero_price": round((df["UnitPrice"] == 0).mean() * 100, 2),
    }

    logger.info(f"  Harga nol: {report['n_zero_price']:,} ({report['pct_zero_price']:.1f}%)")
    logger.info(f"  Harga negatif: {report['n_negative_price']:,}")
    logger.info(f"  Harga min: {report['price_min']:.4f}")
    logger.info(f"  Harga max: {report['price_max']:,.2f}")
    return report


def audit_customer_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    """Statistik cakupan pelanggan."""
    logger.info("Menjalankan audit: cakupan pelanggan")

    valid_customers = df.dropna(subset=["CustomerID"])
    tx_per_customer = valid_customers.groupby("CustomerID")["InvoiceNo"].nunique()

    report = {
        "n_unique_customers_with_id": int(valid_customers["CustomerID"].nunique()),
        "tx_per_customer_mean": round(float(tx_per_customer.mean()), 2),
        "tx_per_customer_median": float(tx_per_customer.median()),
        "tx_per_customer_max": int(tx_per_customer.max()),
        "pct_customers_single_tx": round(
            (tx_per_customer == 1).mean() * 100, 2
        ),
        "pct_customers_ge2_tx": round(
            (tx_per_customer >= 2).mean() * 100, 2
        ),
        "n_unique_countries": int(df["Country"].nunique()),
        "top_countries": df["Country"].value_counts().head(5).to_dict(),
    }

    logger.info(f"  Pelanggan unik (dengan ID): {report['n_unique_customers_with_id']:,}")
    logger.info(f"  Rata-rata transaksi per pelanggan: {report['tx_per_customer_mean']:.1f}")
    logger.info(f"  Pelanggan dengan 1 transaksi: {report['pct_customers_single_tx']:.1f}%")
    logger.info(f"  Negara unik: {report['n_unique_countries']}")
    return report


def audit_revenue_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Kalkulasi revenue gross dan statistik nilai transaksi."""
    logger.info("Menjalankan audit: tinjauan revenue")

    valid = df[
        ~df["InvoiceNo"].str.startswith("C", na=False)
        & df["CustomerID"].notna()
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
    ].copy()

    valid["LineValue"] = valid["Quantity"] * valid["UnitPrice"]

    report = {
        "total_gross_revenue": round(float(valid["LineValue"].sum()), 2),
        "n_valid_line_items": int(len(valid)),
        "revenue_per_line_mean": round(float(valid["LineValue"].mean()), 2),
        "revenue_per_line_median": round(float(valid["LineValue"].median()), 2),
        "revenue_per_line_p99": round(float(valid["LineValue"].quantile(0.99)), 2),
        "revenue_per_line_max": round(float(valid["LineValue"].max()), 2),
        "n_extreme_lines_gt10k": int((valid["LineValue"] > 10000).sum()),
    }

    logger.info(f"  Total gross revenue: £{report['total_gross_revenue']:,.2f}")
    logger.info(f"  Line items valid: {report['n_valid_line_items']:,}")
    logger.info(f"  Revenue per baris (median): £{report['revenue_per_line_median']:.2f}")
    logger.info(f"  Baris dengan revenue > £10k: {report['n_extreme_lines_gt10k']:,}")
    return report


def run_full_audit(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Jalankan semua audit dan kembalikan laporan lengkap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame raw dari loader.load_raw_data().

    Returns
    -------
    dict
        Laporan audit lengkap dengan semua temuan.
    """
    logger.info("=" * 60)
    logger.info("MEMULAI AUDIT DATA LENGKAP")
    logger.info("=" * 60)

    audit_report = {
        "basic_info": audit_basic_info(df),
        "missing_values": audit_missing_values(df),
        "duplicates": audit_duplicates(df),
        "date_range": audit_date_range(df),
        "cancelled_invoices": audit_cancelled_invoices(df),
        "quantity_issues": audit_quantity_issues(df),
        "price_issues": audit_price_issues(df),
        "customer_coverage": audit_customer_coverage(df),
        "revenue_overview": audit_revenue_overview(df),
    }

    logger.info("=" * 60)
    logger.info("AUDIT SELESAI")
    logger.info("=" * 60)

    return audit_report
