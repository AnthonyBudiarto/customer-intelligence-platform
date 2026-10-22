"""
Halaman 5: Methodology — Desain temporal, definisi, limitasi
"""
import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Methodology", layout="wide")
st.title("📖 Metodologi & Limitasi")

ROOT = Path(__file__).parents[2]

st.markdown("""
## 1. Desain Temporal

Platform ini menggunakan **chronological validation** — informasi masa depan tidak pernah digunakan
untuk membangun fitur atau target historis.

```
Rentang Dataset:   01 Des 2010  ──────────────────────────  09 Des 2011
                                                             (373 hari)

Train Split:       01 Des 2010 ─── cutoff 30 Jun 2011 ─── horizon end 08 Sep 2011
                   [─── observation (7 bln) ───][── horizon 70 hari ──]

Test Split:        01 Des 2010 ─── cutoff 30 Sep 2011 ─── horizon end 08 Des 2011
                   [── observation (10 bln) ──────────][── horizon 70 hari ──]
```

**Mengapa 70 hari, bukan 90?**
Dataset berakhir 09 Desember 2011. Cutoff 30 September + 90 hari = 29 Desember 2011 — melampaui data.
70 hari dipilih karena memberikan follow-up penuh dan menghasilkan distribusi kelas yang seimbang (36.8% inactive).
""")

st.markdown("---")
st.markdown("""
## 2. Definisi Target Inaktivitas

**Pelanggan berlabel INACTIVE (1) jika:**
- Mereka memiliki ≥1 qualified purchase dalam 90 hari sebelum prediction cutoff (eligibility)
- Mereka **tidak** melakukan qualified purchase dalam 70 hari setelah cutoff

**Qualified Purchase =** transaksi yang memenuhi semua kriteria:
- CustomerID tidak null
- Bukan invoice cancelled (prefix 'C')
- Quantity ≥ 1
- UnitPrice ≥ 0.01 GBP
- Timestamp valid

**Penting:** Label ini **bukan churn yang sesungguhnya**. Pelanggan mungkin:
- Membeli di toko fisik atau saluran lain
- Sedang dalam siklus pembelian musiman
- Kembali setelah lebih dari 70 hari
""")

st.markdown("---")
st.markdown("""
## 3. Feature Engineering

Semua fitur dihitung menggunakan hanya transaksi **pada atau sebelum** prediction cutoff.

| Fitur | Deskripsi | Catatan |
|-------|-----------|---------|
| recency_days | Hari sejak qualified purchase terakhir | Kunci RFM |
| frequency | Jumlah invoice unik | Kunci RFM |
| monetary_total | Total revenue historis | Kunci RFM |
| monetary_mean / std | Rata-rata & variasi nilai order | |
| ipt_mean / std | Rata-rata & variasi gap antar pembelian | 32% missing (pelanggan 1 tx) |
| n_unique_products | Diversitas produk | |
| n_unique_months | Bulan aktif | |
| tenure_days | Lama jadi pelanggan | |
| recent_freq_ratio | Frekuensi 30 hari terakhir / total | Trend indicator |
| spending_trend | Spend 30 hari terakhir / total | Trend indicator |
| cancellation_rate | Rasio order yang dibatalkan | |
| country_UK | Binary: pelanggan UK | |

**Preprocessing:** Kolom skewed di-log-transform (log1p), kemudian StandardScaler dan median imputer
difit HANYA pada training data, lalu diterapkan ke test data.
""")

st.markdown("---")
st.markdown("""
## 4. Model dan Hyperparameter

| Model | Konfigurasi | Class Imbalance |
|-------|-------------|-----------------|
| Rule-Based | Recency threshold dari train set | N/A |
| Logistic Regression | C=1.0, solver=lbfgs | class_weight=balanced |
| Random Forest | 200 trees, max_depth=10 | class_weight=balanced |
| XGBoost | 300 estimators, lr=0.05, depth=6 | scale_pos_weight |
| MLP (PyTorch) | [128,64,32], dropout=0.3, early stopping | pos_weight |

**Threshold selection:** Dipilih dari validation (train) set menggunakan strategi F1-optimal.
Test set tidak pernah digunakan untuk memilih threshold.
""")

st.markdown("---")
st.markdown("""
## 5. Limitasi dan Peringatan

1. **Dataset kecil**: Hanya 4,372 pelanggan teridentifikasi dalam satu tahun data.
   Hasil mungkin tidak stabil pada sub-populasi kecil.

2. **Satu snapshot temporal**: Dataset terlalu pendek untuk multiple independent snapshots.
   Tidak bisa mengukur stabilitas model antar periode.

3. **UK-centric**: 91.4% transaksi dari UK. Model mungkin tidak generalisable ke pelanggan internasional.

4. **Data tahun 2010-2011**: Pola e-commerce telah berubah signifikan. Model ini tidak dirancang
   untuk deployment produksi di retailer modern.

5. **Bukan churn kausal**: Model memprediksi inaktivitas historis berdasarkan pola, bukan sebab.
   Intervensi (kampanye retensi) memerlukan evaluasi kausal terpisah (RCT atau quasi-experiment).

6. **CustomerID hilang 25%**: Seperempat transaksi tidak bisa diatribusikan ke pelanggan.
   Ini mungkin menyebabkan bias seleksi yang tidak terukur.

7. **Kalibrasi**: Probabilitas prediksi mungkin tidak terkalibrasi sempurna.
   Lihat calibration plot di Model Lab sebelum menggunakan probabilitas secara absolut.
""")

# Load audit report untuk statistik cleaning
try:
    with open(ROOT / "artifacts/reports/audit_report.json") as f:
        audit = json.load(f)

    st.markdown("---")
    st.markdown("## 6. Statistik Data Cleaning")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baris Raw", f"{audit['basic_info']['n_rows']:,}")
    col2.metric("Baris Cleaned (Modeling)", "392,688")
    col3.metric("CustomerID Missing", f"{audit['missing_values']['pct_null_customer_id']:.1f}%")
    col4.metric("Cancelled Invoices", f"{audit['cancelled_invoices']['pct_cancelled_rows']:.1f}%")

    st.markdown("""
    Lihat `reports/audit_report.md` dan `reports/data_dictionary.md` untuk dokumentasi lengkap
    setiap keputusan cleaning.
    """)
except Exception:
    pass
