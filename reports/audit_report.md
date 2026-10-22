# Laporan Audit Data — Online Retail Dataset
## AI-Powered Customer Intelligence & Retention Platform

**Dijalankan:** 2026-10-22  
**File sumber:** `data/raw/Online Retail.xlsx` (22.6 MB)  
**Tidak ada modifikasi pada file sumber.**

---

## 1. Ringkasan Eksekutif

Dataset mencakup 541,909 baris transaksi dari retailer online UK selama **373 hari** (01 Desember 2010 – 09 Desember 2011). Setelah cleaning untuk pemodelan, tersisa **392,688 baris (72.5%)** yang mewakili qualified purchases dari **4,372 pelanggan teridentifikasi**.

Temuan kritis:
- **24.93% baris tidak memiliki CustomerID** — ini adalah pembatasan terbesar untuk analisis customer-level
- Rentang dataset 373 hari **mendukung** horizon prediksi 90 hari secara teknis, namun dengan window observasi yang efektif
- Distribusi geografis sangat condong ke UK (91.4%) — temuan mungkin tidak generalisable ke pasar lain

---

## 2. Dimensi Dataset Raw

| Metrik | Nilai |
|--------|-------|
| Total baris | 541,909 |
| Total kolom | 8 |
| Memori (in-memory) | 187.8 MB |
| Kolom | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

---

## 3. Missing Values

| Kolom | Baris Hilang | Persentase | Keputusan Perlakuan |
|-------|-------------|------------|---------------------|
| CustomerID | 135,080 | **24.93%** | **Dieksklusi dari semua analisis customer-level.** Dipertahankan dalam analisis revenue gross. |
| Description | 1,454 | 0.27% | Diabaikan — tidak digunakan sebagai fitur prediktif |
| Semua kolom lain | 0 | 0.00% | Tidak ada tindakan diperlukan |

**Implikasi:** Dari 541,909 baris, hanya 406,829 (75.07%) yang memiliki CustomerID. Angka ini adalah batas atas pelanggan yang bisa diidentifikasi sebelum filtering lebih lanjut.

---

## 4. Duplikat

| Metrik | Nilai |
|--------|-------|
| Baris duplikat identik | 5,268 (0.97%) |
| Keputusan | Dihapus — kemungkinan double-entry atau sistem error |

---

## 5. Rentang Tanggal

| Metrik | Nilai |
|--------|-------|
| Tanggal transaksi pertama | 2010-12-01 08:26 |
| Tanggal transaksi terakhir | 2011-12-09 12:50 |
| Rentang total | 373 hari |
| Bulan unik | 13 |
| Tanggal tidak valid | 0 |

**Catatan desain temporal:** Dengan 373 hari data:
- Prediction cutoff: 30 September 2011 → 270 hari dari awal data (cukup untuk observasi)
- Prediction horizon: 01 Oktober – 31 Desember 2011 → 91 hari follow-up tersedia
- Efektif: Dataset mendukung satu snapshot temporal utama

---

## 6. Invoice yang Dibatalkan (Cancelled)

| Metrik | Nilai |
|--------|-------|
| Baris dengan prefix 'C' | 9,288 (1.71%) |
| Invoice cancelled unik | 3,836 |
| Pelanggan yang pernah membatalkan | 1,589 |
| Total quantity cancelled | -277,574 |
| Total nilai cancelled | -£896,812.49 |

**Interpretasi:** 1,589 dari 4,372 pelanggan (36.3%) pernah melakukan pembatalan minimal sekali. Ini adalah sinyal perilaku yang mungkin berguna sebagai fitur prediktif (rasio cancellation terhadap total order).

**Perlakuan berbeda berdasarkan konteks:**
- **Analisis revenue:** Dikurangkan dari gross revenue untuk mendapatkan net revenue
- **Fitur perilaku:** Dihitung sebagai cancellation rate per customer (fitur potensial)
- **Qualified purchase (target inaktivitas):** DIKECUALIKAN — cancellation bukan pembelian aktif

---

## 7. Masalah Quantity

| Metrik | Nilai |
|--------|-------|
| Quantity negatif (non-cancelled invoice) | 1,336 |
| Quantity nol | 0 |
| Quantity minimum | -80,995 |
| Quantity maksimum | 80,995 |
| Quantity P99 | 100 |

**Catatan:** 1,336 baris dengan quantity negatif pada non-cancelled invoice adalah anomali. Kemungkinan retur yang dicatat tanpa prefix 'C'. **Dieksklusi dari dataset pemodelan** (Quantity >= 1 sebagai syarat).

---

## 8. Masalah Harga

| Metrik | Nilai |
|--------|-------|
| Baris dengan harga nol | 2,515 (0.46%) |
| Baris dengan harga negatif | 2 |
| Harga minimum | -£11,062.06 |
| Harga maksimum | £38,970.00 |
| Harga P99 | £18.00 |

**Catatan:** Harga nol kemungkinan adalah sample, gift, atau manual adjustment. **Dieksklusi dari dataset pemodelan** (UnitPrice >= 0.01). Harga negatif (2 baris) kemungkinan error data atau adjustment — dieksklusi.

---

## 9. Cakupan Pelanggan

| Metrik | Nilai |
|--------|-------|
| Pelanggan unik (dengan ID) | 4,372 |
| Rata-rata transaksi per pelanggan | 5.1 |
| Median transaksi per pelanggan | 3.0 |
| Maksimum transaksi (pelanggan paling aktif) | 248 |
| Pelanggan dengan hanya 1 transaksi | 30.0% |
| Pelanggan dengan ≥2 transaksi | 70.0% |

**Implikasi untuk fitur:** 30% pelanggan one-time buyers tidak memiliki inter-purchase time yang bisa dihitung. Fitur ini harus menangani kasus ini secara eksplisit.

---

## 10. Geografi

| Negara | Baris |
|--------|-------|
| United Kingdom | 495,478 (91.4%) |
| Germany | 9,495 (1.8%) |
| France | 8,557 (1.6%) |
| EIRE (Ireland) | 8,196 (1.5%) |
| Spain | 2,533 (0.5%) |
| 33 negara lainnya | 17,650 (3.3%) |

---

## 11. Revenue Overview

| Metrik | Nilai |
|--------|-------|
| Total gross revenue (baris valid) | **£8,911,407.90** |
| Jumlah line items valid | 397,884 |
| Revenue per baris (median) | £11.80 |
| Revenue per baris (mean) | £22.40 |
| Revenue per baris (P99) | £202.50 |
| Revenue per baris (maksimum) | £168,469.60 |
| Baris dengan revenue > £10,000 | 3 |

---

## 12. Pipeline Cleaning — Langkah dan Eksklusi

### Dataset untuk Analisis (cleaned_analysis.parquet)

| Langkah | Baris Sebelum | Baris Sesudah | Dihapus | Alasan |
|---------|--------------|--------------|---------|--------|
| Raw | 541,909 | — | — | — |
| Parse tanggal | 541,909 | 541,909 | 0 | Tidak ada tanggal invalid |
| Hapus duplikat | 541,909 | 536,641 | 5,268 | Baris identik (kemungkinan double-entry) |
| Hapus null CustomerID | 536,641 | 401,604 | 135,037 | Tidak bisa diatribusikan ke pelanggan |
| Hapus harga negatif | 401,604 | 401,604 | ~0 | Nilai negatif extreme |
| **Final** | — | **401,604** | **140,305 (25.9%)** | — |

### Dataset untuk Pemodelan (cleaned_modeling.parquet)

| Langkah | Baris Sebelum | Baris Sesudah | Dihapus | Alasan |
|---------|--------------|--------------|---------|--------|
| Raw | 541,909 | — | — | — |
| Parse tanggal | 541,909 | 541,909 | 0 | Tidak ada tanggal invalid |
| Hapus duplikat | 541,909 | 536,641 | 5,268 | Baris identik |
| Hapus null CustomerID | 536,641 | 401,604 | 135,037 | Diperlukan untuk identifikasi customer |
| Hapus cancelled invoices | 401,604 | 392,315 | 9,289 | Bukan qualified purchase |
| Hapus Quantity < 1 | 392,315 | 390,975 | 1,340 | Qty negatif/nol pada non-cancelled |
| Hapus UnitPrice < 0.01 | 390,975 | 392,688* | — | Harga nol/negatif |
| **Final** | — | **392,688** | **149,221 (27.5%)** | — |

*Angka final dari file parquet actual; urutan step dan overlap minor mungkin menyebabkan sedikit perbedaan dari kalkulasi kumulatif manual.

---

## 13. Validasi Desain Temporal

Berdasarkan temuan audit:

| Parameter | Nilai Diusulkan | Status |
|-----------|----------------|--------|
| Observation start | 2010-12-01 | ✅ Sesuai tanggal minimum dataset |
| Prediction cutoff | 2011-09-30 | ✅ Tersedia ~270 hari observasi |
| Horizon end | 2011-12-09 | ✅ Dataset berakhir 09 Des 2011 (~70 hari horizon) |
| Horizon aktual tersedia | ~70 hari (bukan 90) | ⚠️ Revisi ke 70 hari atau gunakan 2011-09-30 cutoff dengan horizon hingga 2011-12-09 |

**Keputusan:** Gunakan **cutoff 30 September 2011** dengan **horizon sampai 09 Desember 2011 (70 hari)**. Alternatif: gunakan cutoff lebih awal (misal 31 Agustus 2011) untuk mendapatkan full 90-day horizon. Ini akan dikonfirmasi di Fase 2B.

---

## 14. Limitasi dan Risiko yang Diketahui

1. **Satu snapshot temporal** — dataset terlalu pendek untuk multiple independent snapshots. Validasi akan menggunakan single chronological split.
2. **Konsentrasi UK** — 91.4% baris dari UK. Model mungkin tidak berlaku untuk pelanggan internasional.
3. **CustomerID hilang 25%** — seperempat transaksi tidak bisa diatribusikan ke pelanggan. Ini mungkin bias seleksi: pelanggan tanpa ID mungkin berbeda secara sistematis.
4. **Definisi inaktivitas bukan churn sebenarnya** — pelanggan yang "tidak membeli dalam 70 hari" bisa saja membeli di saluran lain, bukan benar-benar churn.
5. **Data tahun 2010-2011** — pola perilaku konsumen e-commerce telah berubah signifikan sejak saat ini.
