"""
Modul untuk memuat dataset raw Online Retail dari file Excel.

Tidak melakukan cleaning di sini — hanya loading dan validasi schema dasar.
Semua cleaning dilakukan di cleaner.py.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("data.loader")

# Kolom yang diharapkan ada di dataset UCI Online Retail
EXPECTED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
}


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Muat dataset Online Retail dari file Excel.

    File dibaca apa adanya — tidak ada filtering, tidak ada transformasi.
    Tujuannya adalah mendapatkan snapshot yang identik dengan file sumber.

    Parameters
    ----------
    filepath : str, optional
        Path ke file xlsx. Jika None, diambil dari configs/project_config.yaml.

    Returns
    -------
    pd.DataFrame
        Raw dataframe dengan semua kolom dan baris asli.

    Raises
    ------
    FileNotFoundError
        Jika file tidak ditemukan di path yang ditentukan.
    ValueError
        Jika kolom yang diharapkan tidak ada dalam dataset.
    """
    if filepath is None:
        config = load_config()
        filepath = resolve_path(config["paths"]["raw_data"])
    else:
        filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {filepath}\n"
            f"Unduh dari: https://archive.uci.edu/dataset/352/online+retail\n"
            f"Simpan sebagai: data/raw/Online Retail.xlsx"
        )

    logger.info(f"Memuat dataset dari: {filepath}")
    logger.info(f"Ukuran file: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

    # Coba baca dengan engine openpyxl (lebih andal untuk .xlsx)
    try:
        df = pd.read_excel(
            filepath,
            engine="openpyxl",
            dtype={
                "InvoiceNo": str,
                "StockCode": str,
                "CustomerID": str,  # baca sebagai string dulu, konversi nanti
                "Country": str,
                "Description": str,
            },
        )
    except Exception as e:
        logger.error(f"Gagal membaca file Excel: {e}")
        raise

    logger.info(f"Dataset dimuat: {df.shape[0]:,} baris × {df.shape[1]} kolom")

    # Validasi kolom
    actual_cols = set(df.columns)
    missing_cols = EXPECTED_COLUMNS - actual_cols
    extra_cols = actual_cols - EXPECTED_COLUMNS

    if missing_cols:
        raise ValueError(
            f"Kolom yang diharapkan tidak ditemukan dalam dataset: {missing_cols}\n"
            f"Kolom aktual: {sorted(actual_cols)}"
        )

    if extra_cols:
        logger.warning(f"Kolom tambahan yang tidak diharapkan: {extra_cols}")

    # Strip whitespace dari kolom string
    for col in ["InvoiceNo", "StockCode", "Description", "Country", "CustomerID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Ganti string 'nan' (dari konversi object → str) kembali ke NaN
    for col in ["CustomerID", "Description"]:
        if col in df.columns:
            df[col] = df[col].replace("nan", pd.NA)

    logger.info("Schema validasi: OK")
    return df


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Ringkasan cepat tentang dataset yang dimuat.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame raw dari load_raw_data().

    Returns
    -------
    dict
        Dictionary berisi statistik dasar dataset.
    """
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
        "date_min": None,
        "date_max": None,
        "n_unique_invoices": None,
        "n_unique_customers": None,
        "n_unique_products": None,
        "n_missing_customer_id": None,
    }

    if "InvoiceDate" in df.columns:
        dates = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        info["date_min"] = dates.min()
        info["date_max"] = dates.max()

    if "InvoiceNo" in df.columns:
        info["n_unique_invoices"] = df["InvoiceNo"].nunique()

    if "CustomerID" in df.columns:
        info["n_missing_customer_id"] = df["CustomerID"].isna().sum()
        info["n_unique_customers"] = df["CustomerID"].dropna().nunique()

    if "StockCode" in df.columns:
        info["n_unique_products"] = df["StockCode"].nunique()

    return info
