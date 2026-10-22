"""
Pipeline pembersihan data Online Retail.

Prinsip utama:
- File raw TIDAK pernah dimodifikasi.
- Setiap eksklusi dicatat dengan alasan yang jelas.
- Aturan berbeda diterapkan untuk analitik vs. pemodelan prediktif.
- Fungsi ini deterministik: input yang sama menghasilkan output yang sama.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("data.cleaner")


class CleaningLog:
    """Catat semua eksklusi dan transformasi secara terstruktur."""

    def __init__(self, initial_rows: int):
        self.initial_rows = initial_rows
        self.steps = []
        self.current_rows = initial_rows

    def record(self, step: str, rows_before: int, rows_after: int, reason: str):
        removed = rows_before - rows_after
        pct = removed / self.initial_rows * 100 if self.initial_rows > 0 else 0
        self.steps.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": removed,
                "pct_of_original": round(pct, 3),
                "reason": reason,
            }
        )
        self.current_rows = rows_after
        logger.info(
            f"  [{step}] Hapus {removed:,} baris ({pct:.2f}%) → tersisa {rows_after:,}"
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def summary(self) -> dict:
        return {
            "initial_rows": self.initial_rows,
            "final_rows": self.current_rows,
            "total_removed": self.initial_rows - self.current_rows,
            "pct_retained": round(self.current_rows / self.initial_rows * 100, 2),
            "steps": self.steps,
        }


def clean_for_analysis(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, CleaningLog]:
    """
    Bersihkan dataset untuk analisis customer dan revenue.

    Mencakup invoice yang dibatalkan sebagai informasi perilaku (return behavior),
    tapi mengeksklusi baris yang tidak bisa diidentifikasi.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe dari loader.
    config : dict, optional
        Konfigurasi proyek. Dimuat otomatis jika None.

    Returns
    -------
    Tuple[pd.DataFrame, CleaningLog]
        DataFrame bersih dan log eksklusi.
    """
    if config is None:
        config = load_config()

    cfg = config["cleaning"]
    log = CleaningLog(len(df))

    logger.info("Memulai cleaning untuk ANALISIS...")
    df = df.copy()

    # --- Step 1: Parse tanggal ---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    n_before = len(df)
    df = df[df["InvoiceDate"].notna()]
    log.record("parse_date", n_before, len(df), "Hapus baris dengan tanggal tidak valid")

    # --- Step 2: Hapus duplikat baris identik ---
    n_before = len(df)
    df = df.drop_duplicates()
    log.record("remove_duplicates", n_before, len(df), "Hapus duplikat baris identik")

    # --- Step 3: Hapus baris tanpa CustomerID ---
    n_before = len(df)
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "")]
    log.record(
        "remove_null_customerid",
        n_before,
        len(df),
        "Hapus baris tanpa CustomerID (tidak bisa diatribusikan ke pelanggan)",
    )

    # --- Step 4: Hapus UnitPrice negatif ---
    n_before = len(df)
    df = df[df["UnitPrice"] >= 0]
    log.record("remove_negative_price", n_before, len(df), "Hapus baris dengan UnitPrice negatif")

    # Tambahkan kolom turunan berguna
    df["IsCancelled"] = df["InvoiceNo"].str.startswith("C", na=False)
    df["LineValue"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

    logger.info(f"Cleaning analisis selesai: {len(df):,} baris dipertahankan")
    return df, log


def clean_for_modeling(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, CleaningLog]:
    """
    Bersihkan dataset untuk pemodelan prediktif.

    Lebih ketat dibanding clean_for_analysis:
    - Hanya transaksi pembelian yang valid (bukan cancellation).
    - Quantity > 0 dan UnitPrice > 0.
    - Ini digunakan untuk membangun fitur RFM dan target inaktivitas.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe dari loader (bukan dari clean_for_analysis).
    config : dict, optional
        Konfigurasi proyek.

    Returns
    -------
    Tuple[pd.DataFrame, CleaningLog]
        DataFrame bersih untuk modeling dan log eksklusi.
    """
    if config is None:
        config = load_config()

    cfg = config["cleaning"]
    log = CleaningLog(len(df))

    logger.info("Memulai cleaning untuk PEMODELAN...")
    df = df.copy()

    # --- Step 1: Parse tanggal ---
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    n_before = len(df)
    df = df[df["InvoiceDate"].notna()]
    log.record("parse_date", n_before, len(df), "Hapus baris dengan tanggal tidak valid")

    # --- Step 2: Hapus duplikat ---
    n_before = len(df)
    df = df.drop_duplicates()
    log.record("remove_duplicates", n_before, len(df), "Hapus duplikat baris identik")

    # --- Step 3: Hapus baris tanpa CustomerID ---
    n_before = len(df)
    df = df[df["CustomerID"].notna() & (df["CustomerID"] != "")]
    log.record(
        "remove_null_customerid",
        n_before,
        len(df),
        "Hapus baris tanpa CustomerID — tidak bisa digunakan untuk fitur customer-level",
    )

    # --- Step 4: Hapus cancelled invoices ---
    n_before = len(df)
    df = df[~df["InvoiceNo"].str.startswith(cfg["cancelled_invoice_prefix"], na=False)]
    log.record(
        "remove_cancelled_invoices",
        n_before,
        len(df),
        f"Hapus invoice dengan prefix '{cfg['cancelled_invoice_prefix']}' (cancelled/credit notes) — "
        "tidak merepresentasikan qualified purchase untuk definisi target inaktivitas",
    )

    # --- Step 5: Hapus Quantity <= 0 ---
    n_before = len(df)
    df = df[df["Quantity"] >= cfg["min_quantity"]]
    log.record(
        "remove_invalid_quantity",
        n_before,
        len(df),
        f"Hapus baris dengan Quantity < {cfg['min_quantity']} pada non-cancelled invoice",
    )

    # --- Step 6: Hapus UnitPrice <= 0 ---
    n_before = len(df)
    df = df[df["UnitPrice"] >= cfg["min_unit_price"]]
    log.record(
        "remove_invalid_price",
        n_before,
        len(df),
        f"Hapus baris dengan UnitPrice < {cfg['min_unit_price']} — "
        "kemungkinan test records, samples, atau manual adjustments",
    )

    # Kolom turunan
    df["LineValue"] = df["Quantity"] * df["UnitPrice"]
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")

    logger.info(f"Cleaning pemodelan selesai: {len(df):,} baris dipertahankan")
    return df, log


def save_cleaned_data(
    df_analysis: pd.DataFrame,
    df_modeling: pd.DataFrame,
    log_analysis: CleaningLog,
    log_modeling: CleaningLog,
    config: dict = None,
) -> Dict[str, Path]:
    """
    Simpan kedua versi cleaned data dan cleaning logs ke disk.

    Returns
    -------
    dict
        Dictionary path ke file yang disimpan.
    """
    if config is None:
        config = load_config()

    from src.utils.config import get_project_root
    import json

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_data"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Simpan cleaned data
    path_analysis = processed_dir / "cleaned_analysis.parquet"
    path_modeling = processed_dir / "cleaned_modeling.parquet"
    path_log_analysis = processed_dir / "cleaning_log_analysis.csv"
    path_log_modeling = processed_dir / "cleaning_log_modeling.csv"
    path_summary = processed_dir / "cleaning_summary.json"

    df_analysis.to_parquet(path_analysis, index=False)
    df_modeling.to_parquet(path_modeling, index=False)
    log_analysis.to_dataframe().to_csv(path_log_analysis, index=False)
    log_modeling.to_dataframe().to_csv(path_log_modeling, index=False)

    # Simpan summary JSON
    summary = {
        "analysis": log_analysis.summary(),
        "modeling": log_modeling.summary(),
    }

    with open(path_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Data analysis disimpan: {path_analysis}")
    logger.info(f"Data modeling disimpan: {path_modeling}")
    logger.info(f"Cleaning summary disimpan: {path_summary}")

    return {
        "analysis": path_analysis,
        "modeling": path_modeling,
        "log_analysis": path_log_analysis,
        "log_modeling": path_log_modeling,
        "summary": path_summary,
    }
