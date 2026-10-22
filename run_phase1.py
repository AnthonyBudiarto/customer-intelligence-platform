"""
Script Phase 1 sederhana â€” jalankan dari root proyek dengan python langsung.
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Memuat library...")
import pandas as pd
import numpy as np
import yaml

print("Memuat konfigurasi...")
with open("configs/project_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================
print("\n[1/4] Memuat dataset raw (ini bisa memakan 30-60 detik)...")
raw_path = Path("data/raw/Online Retail.xlsx")
df = pd.read_excel(
    raw_path,
    engine="openpyxl",
    dtype={"InvoiceNo": str, "StockCode": str, "CustomerID": str, "Country": str, "Description": str},
)
print(f"  Dataset dimuat: {df.shape[0]:,} baris Ã— {df.shape[1]} kolom")

# Strip whitespace
for col in ["InvoiceNo", "StockCode", "Description", "Country", "CustomerID"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
for col in ["CustomerID", "Description"]:
    df[col] = df[col].replace("nan", pd.NA)

# ============================================================
# STEP 2: AUDIT
# ============================================================
print("\n[2/4] Menjalankan audit...")

# Parse tanggal
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

audit = {}

# Basic info
audit["basic_info"] = {
    "n_rows": int(len(df)),
    "n_columns": int(len(df.columns)),
    "columns": list(df.columns),
    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
}

# Missing values
missing_count = df.isnull().sum()
audit["missing_values"] = {
    "missing_count": {k: int(v) for k, v in missing_count.items()},
    "missing_pct": {k: round(v / len(df) * 100, 2) for k, v in missing_count.items()},
    "rows_with_null_customer_id": int(df["CustomerID"].isnull().sum()),
    "pct_null_customer_id": round(df["CustomerID"].isnull().mean() * 100, 2),
}

# Duplicates
audit["duplicates"] = {
    "n_duplicate_rows": int(df.duplicated().sum()),
    "pct_duplicate_rows": round(df.duplicated().mean() * 100, 2),
}

# Date range
audit["date_range"] = {
    "date_min": str(df["InvoiceDate"].min()),
    "date_max": str(df["InvoiceDate"].max()),
    "date_range_days": int((df["InvoiceDate"].max() - df["InvoiceDate"].min()).days),
    "n_unique_months": int(df["InvoiceDate"].dt.to_period("M").nunique()),
    "n_invalid_dates": int(df["InvoiceDate"].isnull().sum()),
}

# Cancelled invoices
is_cancelled = df["InvoiceNo"].str.startswith("C", na=False)
cancelled_df = df[is_cancelled]
audit["cancelled_invoices"] = {
    "n_cancelled_rows": int(is_cancelled.sum()),
    "pct_cancelled_rows": round(is_cancelled.mean() * 100, 2),
    "n_cancelled_invoices": int(cancelled_df["InvoiceNo"].nunique()),
    "n_customers_with_cancellations": int(cancelled_df["CustomerID"].dropna().nunique()),
    "total_cancelled_quantity": int(cancelled_df["Quantity"].sum()),
    "total_cancelled_value": round(float((cancelled_df["Quantity"] * cancelled_df["UnitPrice"]).sum()), 2),
}

# Quantity issues (non-cancelled)
non_cancelled = df[~is_cancelled]
audit["quantity_issues"] = {
    "n_negative_quantity_non_cancelled": int((non_cancelled["Quantity"] < 0).sum()),
    "n_zero_quantity": int((non_cancelled["Quantity"] == 0).sum()),
    "quantity_min": float(df["Quantity"].min()),
    "quantity_max": float(df["Quantity"].max()),
    "quantity_p99": float(df["Quantity"].quantile(0.99)),
}

# Price issues
audit["price_issues"] = {
    "n_zero_price": int((df["UnitPrice"] == 0).sum()),
    "n_negative_price": int((df["UnitPrice"] < 0).sum()),
    "price_min": float(df["UnitPrice"].min()),
    "price_max": float(df["UnitPrice"].max()),
    "price_p99": float(df["UnitPrice"].quantile(0.99)),
}

# Customer coverage
valid_customers = df.dropna(subset=["CustomerID"])
tx_per_customer = valid_customers.groupby("CustomerID")["InvoiceNo"].nunique()
audit["customer_coverage"] = {
    "n_unique_customers_with_id": int(valid_customers["CustomerID"].nunique()),
    "tx_per_customer_mean": round(float(tx_per_customer.mean()), 2),
    "tx_per_customer_median": float(tx_per_customer.median()),
    "tx_per_customer_max": int(tx_per_customer.max()),
    "pct_customers_single_tx": round((tx_per_customer == 1).mean() * 100, 2),
    "pct_customers_ge2_tx": round((tx_per_customer >= 2).mean() * 100, 2),
    "n_unique_countries": int(df["Country"].nunique()),
    "top_countries": {k: int(v) for k, v in df["Country"].value_counts().head(5).items()},
}

# Revenue overview
valid = df[
    ~is_cancelled
    & df["CustomerID"].notna()
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
].copy()
valid["LineValue"] = valid["Quantity"] * valid["UnitPrice"]
audit["revenue_overview"] = {
    "total_gross_revenue": round(float(valid["LineValue"].sum()), 2),
    "n_valid_line_items": int(len(valid)),
    "revenue_per_line_mean": round(float(valid["LineValue"].mean()), 2),
    "revenue_per_line_median": round(float(valid["LineValue"].median()), 2),
    "revenue_per_line_p99": round(float(valid["LineValue"].quantile(0.99)), 2),
    "revenue_per_line_max": round(float(valid["LineValue"].max()), 2),
    "n_extreme_lines_gt10k": int((valid["LineValue"] > 10000).sum()),
}

# ============================================================
# STEP 3: CLEANING
# ============================================================
print("\n[3/4] Menjalankan cleaning pipeline...")

# --- Cleaning untuk ANALISIS ---
df_a = df.copy()
log_a_steps = []

n = len(df_a)
df_a = df_a[df_a["InvoiceDate"].notna()]
log_a_steps.append({"step": "parse_date", "rows_before": n, "rows_after": len(df_a), "rows_removed": n - len(df_a)})

n = len(df_a)
df_a = df_a.drop_duplicates()
log_a_steps.append({"step": "remove_duplicates", "rows_before": n, "rows_after": len(df_a), "rows_removed": n - len(df_a)})

n = len(df_a)
df_a = df_a[df_a["CustomerID"].notna() & (df_a["CustomerID"] != "")]
log_a_steps.append({"step": "remove_null_customerid", "rows_before": n, "rows_after": len(df_a), "rows_removed": n - len(df_a)})

n = len(df_a)
df_a = df_a[df_a["UnitPrice"] >= 0]
log_a_steps.append({"step": "remove_negative_price", "rows_before": n, "rows_after": len(df_a), "rows_removed": n - len(df_a)})

df_a["IsCancelled"] = df_a["InvoiceNo"].str.startswith("C", na=False)
df_a["LineValue"] = df_a["Quantity"] * df_a["UnitPrice"]
df_a["YearMonth"] = df_a["InvoiceDate"].dt.to_period("M")

print(f"  Cleaning analisis: {df.shape[0]:,} â†’ {len(df_a):,} baris")

# --- Cleaning untuk PEMODELAN ---
df_m = df.copy()
log_m_steps = []

n = len(df_m)
df_m = df_m[df_m["InvoiceDate"].notna()]
log_m_steps.append({"step": "parse_date", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

n = len(df_m)
df_m = df_m.drop_duplicates()
log_m_steps.append({"step": "remove_duplicates", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

n = len(df_m)
df_m = df_m[df_m["CustomerID"].notna() & (df_m["CustomerID"] != "")]
log_m_steps.append({"step": "remove_null_customerid", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

n = len(df_m)
df_m = df_m[~df_m["InvoiceNo"].str.startswith("C", na=False)]
log_m_steps.append({"step": "remove_cancelled", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

n = len(df_m)
df_m = df_m[df_m["Quantity"] >= 1]
log_m_steps.append({"step": "remove_invalid_quantity", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

n = len(df_m)
df_m = df_m[df_m["UnitPrice"] >= 0.01]
log_m_steps.append({"step": "remove_invalid_price", "rows_before": n, "rows_after": len(df_m), "rows_removed": n - len(df_m)})

df_m["LineValue"] = df_m["Quantity"] * df_m["UnitPrice"]
df_m["YearMonth"] = df_m["InvoiceDate"].dt.to_period("M")

print(f"  Cleaning pemodelan: {df.shape[0]:,} â†’ {len(df_m):,} baris")

# ============================================================
# STEP 4: SIMPAN
# ============================================================
print("\n[4/4] Menyimpan hasil...")

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("artifacts/reports").mkdir(parents=True, exist_ok=True)

df_a.to_parquet("data/processed/cleaned_analysis.parquet", index=False)
df_m.to_parquet("data/processed/cleaned_modeling.parquet", index=False)

pd.DataFrame(log_a_steps).to_csv("data/processed/cleaning_log_analysis.csv", index=False)
pd.DataFrame(log_m_steps).to_csv("data/processed/cleaning_log_modeling.csv", index=False)

cleaning_summary = {
    "analysis": {
        "initial_rows": int(df.shape[0]),
        "final_rows": int(len(df_a)),
        "pct_retained": round(len(df_a) / df.shape[0] * 100, 2),
        "steps": log_a_steps,
    },
    "modeling": {
        "initial_rows": int(df.shape[0]),
        "final_rows": int(len(df_m)),
        "pct_retained": round(len(df_m) / df.shape[0] * 100, 2),
        "steps": log_m_steps,
    },
}
with open("data/processed/cleaning_summary.json", "w") as f:
    json.dump(cleaning_summary, f, indent=2, default=str)

with open("artifacts/reports/audit_report.json", "w") as f:
    json.dump(audit, f, indent=2, default=str)

# ============================================================
# PRINT RINGKASAN
# ============================================================
print("\n" + "=" * 65)
print("RINGKASAN AUDIT & CLEANING")
print("=" * 65)
print(f"\nðŸ“Š DATASET RAW:")
print(f"  Baris total       : {audit['basic_info']['n_rows']:,}")
print(f"  Kolom             : {audit['basic_info']['n_columns']}")
print(f"  Memori            : {audit['basic_info']['memory_mb']:.1f} MB")

print(f"\nðŸ“… RENTANG WAKTU:")
print(f"  Tanggal awal      : {audit['date_range']['date_min']}")
print(f"  Tanggal akhir     : {audit['date_range']['date_max']}")
print(f"  Rentang (hari)    : {audit['date_range']['date_range_days']}")
print(f"  Bulan unik        : {audit['date_range']['n_unique_months']}")

print(f"\nâŒ MISSING VALUES:")
print(f"  CustomerID hilang : {audit['missing_values']['rows_with_null_customer_id']:,} ({audit['missing_values']['pct_null_customer_id']:.1f}%)")
print(f"  Description hilang: {audit['missing_values']['missing_count']['Description']:,}")

print(f"\nðŸ”„ CANCELLED INVOICES:")
print(f"  Baris cancelled   : {audit['cancelled_invoices']['n_cancelled_rows']:,} ({audit['cancelled_invoices']['pct_cancelled_rows']:.1f}%)")
print(f"  Invoice unik      : {audit['cancelled_invoices']['n_cancelled_invoices']:,}")
print(f"  Nilai cancelled   : Â£{audit['cancelled_invoices']['total_cancelled_value']:,.2f}")

print(f"\nðŸ‘¥ PELANGGAN:")
print(f"  Pelanggan unik    : {audit['customer_coverage']['n_unique_customers_with_id']:,}")
print(f"  Transaksi/customer: {audit['customer_coverage']['tx_per_customer_mean']:.1f} (rata-rata)")
print(f"  1 transaksi saja  : {audit['customer_coverage']['pct_customers_single_tx']:.1f}%")
print(f"  â‰¥2 transaksi      : {audit['customer_coverage']['pct_customers_ge2_tx']:.1f}%")
print(f"  Negara unik       : {audit['customer_coverage']['n_unique_countries']}")
print(f"  Top 5 negara      : {audit['customer_coverage']['top_countries']}")

print(f"\nðŸ’° REVENUE (data valid):")
print(f"  Total gross rev   : Â£{audit['revenue_overview']['total_gross_revenue']:,.2f}")
print(f"  Median per baris  : Â£{audit['revenue_overview']['revenue_per_line_median']:.2f}")
print(f"  Max per baris     : Â£{audit['revenue_overview']['revenue_per_line_max']:,.2f}")
print(f"  Baris > Â£10k      : {audit['revenue_overview']['n_extreme_lines_gt10k']:,}")

print(f"\nâš™ï¸  QUANTITY & PRICE:")
print(f"  Qty negatif (non-C): {audit['quantity_issues']['n_negative_quantity_non_cancelled']:,}")
print(f"  Harga nol         : {audit['price_issues']['n_zero_price']:,} ({round(audit['price_issues']['n_zero_price']/audit['basic_info']['n_rows']*100,2):.1f}%)")
print(f"  Duplikat baris    : {audit['duplicates']['n_duplicate_rows']:,} ({audit['duplicates']['pct_duplicate_rows']:.1f}%)")

print(f"\nðŸ§¹ HASIL CLEANING:")
for mode, summ in cleaning_summary.items():
    print(f"  [{mode.upper():10s}] {summ['initial_rows']:,} â†’ {summ['final_rows']:,} baris ({summ['pct_retained']:.1f}% dipertahankan)")

print(f"\nðŸ“ FILE TERSIMPAN:")
print(f"  data/processed/cleaned_analysis.parquet")
print(f"  data/processed/cleaned_modeling.parquet")
print(f"  data/processed/cleaning_log_analysis.csv")
print(f"  data/processed/cleaning_log_modeling.csv")
print(f"  data/processed/cleaning_summary.json")
print(f"  artifacts/reports/audit_report.json")
print("\nâœ… FASE 1 SELESAI")

