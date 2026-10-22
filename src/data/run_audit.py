"""
Script utama untuk menjalankan audit data dan cleaning pipeline.

Jalankan dari root proyek:
    python -m src.data.run_audit

Output:
    - data/processed/cleaned_analysis.parquet
    - data/processed/cleaned_modeling.parquet
    - data/processed/cleaning_log_analysis.csv
    - data/processed/cleaning_log_modeling.csv
    - data/processed/cleaning_summary.json
    - artifacts/reports/audit_report.json
"""

import json
import sys
from pathlib import Path

# Tambahkan root proyek ke sys.path agar import src.* berjalan
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data, get_dataset_info
from src.data.auditor import run_full_audit
from src.data.cleaner import clean_for_analysis, clean_for_modeling, save_cleaned_data
from src.utils.config import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger("run_audit")


def main():
    logger.info("=" * 70)
    logger.info("FASE 1: DATA AUDIT & CLEANING")
    logger.info("AI-Powered Customer Intelligence & Retention Platform")
    logger.info("=" * 70)

    config = load_config()
    root = get_project_root()

    # -------------------------------------------------------------------------
    # 1. LOAD RAW DATA
    # -------------------------------------------------------------------------
    logger.info("\n[1/4] Memuat dataset raw...")
    df_raw = load_raw_data()
    info = get_dataset_info(df_raw)

    logger.info(f"  Baris: {info['n_rows']:,}")
    logger.info(f"  Kolom: {info['n_columns']}")
    logger.info(f"  Tanggal awal: {info['date_min']}")
    logger.info(f"  Tanggal akhir: {info['date_max']}")
    logger.info(f"  Pelanggan unik: {info['n_unique_customers']:,}")
    logger.info(f"  Invoice unik: {info['n_unique_invoices']:,}")
    logger.info(f"  Produk unik: {info['n_unique_products']:,}")
    logger.info(f"  CustomerID hilang: {info['n_missing_customer_id']:,}")

    # -------------------------------------------------------------------------
    # 2. JALANKAN AUDIT LENGKAP
    # -------------------------------------------------------------------------
    logger.info("\n[2/4] Menjalankan audit lengkap...")
    audit_report = run_full_audit(df_raw)

    # Simpan audit report
    reports_dir = root / config["paths"]["artifacts"] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_path = reports_dir / "audit_report.json"

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, default=str)

    logger.info(f"\nAudit report disimpan: {audit_path}")

    # -------------------------------------------------------------------------
    # 3. CLEANING
    # -------------------------------------------------------------------------
    logger.info("\n[3/4] Menjalankan cleaning pipeline...")

    df_analysis, log_analysis = clean_for_analysis(df_raw, config)
    df_modeling, log_modeling = clean_for_modeling(df_raw, config)

    logger.info(f"\nRingkasan cleaning:")
    logger.info(f"  [ANALISIS]  {log_analysis.initial_rows:,} → {len(df_analysis):,} baris "
                f"(dipertahankan {log_analysis.summary()['pct_retained']:.1f}%)")
    logger.info(f"  [PEMODELAN] {log_modeling.initial_rows:,} → {len(df_modeling):,} baris "
                f"(dipertahankan {log_modeling.summary()['pct_retained']:.1f}%)")

    # -------------------------------------------------------------------------
    # 4. SIMPAN
    # -------------------------------------------------------------------------
    logger.info("\n[4/4] Menyimpan data yang sudah dibersihkan...")
    saved_paths = save_cleaned_data(
        df_analysis, df_modeling, log_analysis, log_modeling, config
    )

    # -------------------------------------------------------------------------
    # RINGKASAN AKHIR
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("AUDIT & CLEANING SELESAI")
    logger.info("=" * 70)

    date_range_days = audit_report["date_range"]["date_range_days"]
    n_cancelled_rows = audit_report["cancelled_invoices"]["n_cancelled_rows"]
    pct_cancelled = audit_report["cancelled_invoices"]["pct_cancelled_rows"]
    n_missing_cid = audit_report["missing_values"]["rows_with_null_customer_id"]
    pct_missing_cid = audit_report["missing_values"]["pct_null_customer_id"]

    logger.info(f"\n📅 Rentang dataset: {audit_report['date_range']['date_min']} "
                f"→ {audit_report['date_range']['date_max']} ({date_range_days} hari)")
    logger.info(f"❌ CustomerID hilang: {n_missing_cid:,} baris ({pct_missing_cid:.1f}%)")
    logger.info(f"🔄 Cancelled invoices: {n_cancelled_rows:,} baris ({pct_cancelled:.1f}%)")
    logger.info(f"💰 Total gross revenue (data bersih): "
                f"£{audit_report['revenue_overview']['total_gross_revenue']:,.2f}")

    logger.info("\n⚠️  CATATAN DESAIN TEMPORAL:")
    if date_range_days < 180:
        logger.warning(
            f"  Dataset hanya mencakup {date_range_days} hari. "
            f"Horizon 90 hari mungkin tidak feasible. Tinjau ulang desain."
        )
    else:
        logger.info(
            f"  Dataset mencakup {date_range_days} hari — "
            f"horizon 90 hari layak dievaluasi lebih lanjut di Fase EDA."
        )

    logger.info("\nFile output:")
    for name, path in saved_paths.items():
        logger.info(f"  {name}: {path}")

    return {
        "df_raw": df_raw,
        "df_analysis": df_analysis,
        "df_modeling": df_modeling,
        "audit_report": audit_report,
        "log_analysis": log_analysis,
        "log_modeling": log_modeling,
    }


if __name__ == "__main__":
    results = main()
