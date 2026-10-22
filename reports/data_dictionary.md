# Data Dictionary — Online Retail Dataset
## AI-Powered Customer Intelligence & Retention Platform

**Sumber:** UCI Machine Learning Repository — Online Retail Dataset  
**URL:** https://archive.uci.edu/dataset/352/online+retail  
**Format file raw:** `data/raw/Online Retail.xlsx`  
**Diaudit pada:** 2026-10-22  
**Baris raw:** 541,909 | **Kolom:** 8

---

## Kolom Asli

| Kolom | Tipe Asli | Tipe Setelah Parse | Deskripsi | Contoh Nilai | Catatan Kualitas |
|-------|-----------|-------------------|-----------|--------------|-----------------|
| `InvoiceNo` | object | str | Nomor faktur. Invoice yang dibatalkan diawali dengan 'C'. | `536365`, `C536379` | Tidak ada missing. 3,836 invoice unik yang dibatalkan. |
| `StockCode` | object | str | Kode produk/stok. Beberapa kode bersifat non-produk (misal: POST, DOT, M, BANK CHARGES). | `85123A`, `POST` | Tidak ada missing. Beberapa kode perlu diperiksa jika digunakan sebagai fitur produk. |
| `Description` | object | str | Deskripsi teks produk. | `WHITE HANGING HEART T-LIGHT HOLDER` | 1,454 missing (0.27%). Tidak digunakan sebagai fitur prediktif utama. |
| `Quantity` | int64 | int64 | Jumlah unit per baris transaksi. Negatif pada invoice cancelled dan beberapa baris anomali non-cancelled. | `6`, `-1`, `80995` | Tidak ada missing. Min: -80,995. Max: 80,995. P99: 100. |
| `InvoiceDate` | datetime | datetime64 | Tanggal dan waktu transaksi. | `2010-12-01 08:26:00` | Tidak ada missing. Rentang: 01 Des 2010 – 09 Des 2011 (373 hari). |
| `UnitPrice` | float64 | float64 | Harga satuan dalam GBP (Pound Sterling). | `2.55`, `0.00`, `38970.00` | 2,515 baris harga nol (0.46%). 2 baris harga negatif. Max: £38,970. |
| `CustomerID` | float64 | str/NA | ID numerik pelanggan. Nilai hilang menunjukkan transaksi tanpa identifikasi pelanggan. | `17850`, `NaN` | **135,080 missing (24.93%).** Baris ini tidak dapat digunakan untuk analisis customer-level. |
| `Country` | object | str | Negara tempat pelanggan berdomisili. | `United Kingdom`, `Germany` | Tidak ada missing. 38 negara unik. UK mendominasi (91.4%). |

---

## Kolom Turunan (Dibuat Saat Cleaning)

| Kolom | Dibuat Di | Deskripsi |
|-------|-----------|-----------|
| `IsCancelled` | `clean_for_analysis()` | Boolean: True jika InvoiceNo dimulai dengan 'C' |
| `LineValue` | Keduanya | `Quantity × UnitPrice` — nilai moneter satu baris transaksi |
| `YearMonth` | Keduanya | Periode bulan (`Period[M]`) untuk agregasi temporal |

---

## Definisi "Qualified Purchase" (Untuk Pemodelan)

Sebuah transaksi dianggap **qualified purchase** dan diikutsertakan dalam dataset pemodelan jika memenuhi **semua** kriteria berikut:

1. `CustomerID` tidak null
2. `InvoiceNo` tidak diawali 'C' (bukan cancellation/credit note)
3. `Quantity >= 1`
4. `UnitPrice >= 0.01`
5. `InvoiceDate` dapat diparsing sebagai datetime yang valid
6. Bukan baris duplikat identik

Transaksi yang tidak memenuhi kriteria ini **tidak dihitung** sebagai bukti aktivitas pelanggan dalam definisi target inaktivitas.

---

## Asumsi Mata Uang

Semua nilai moneter diasumsikan dalam **GBP (Pound Sterling)**. Dataset tidak memiliki kolom mata uang eksplisit. Karena mayoritas transaksi berasal dari UK (91.4%), asumsi ini masuk akal, namun transaksi internasional mungkin menggunakan mata uang lokal yang sudah dikonversi (tidak dapat dikonfirmasi dari data).

---

## Catatan Penting untuk Interview

- **CustomerID hilang 24.93%**: Ini adalah keputusan bisnis yang signifikan. Baris-baris ini kemungkinan adalah pembelian tamu (guest checkout) atau transaksi B2B tanpa ID terdaftar. Mereka dieksklusi dari **semua** analisis customer-level dan pemodelan, tetapi termasuk dalam analisis revenue gross.
- **Invoice 'C'**: Ini adalah retur/pembatalan, bukan pembelian baru. Mereka memiliki nilai `Quantity` negatif. Perlakuan berbeda tergantung konteks: **diikutsertakan** dalam analisis perilaku (untuk mengetahui pola retur), **dieksklusi** dari definisi qualified purchase untuk target inaktivitas.
- **Tidak ada label churn**: Dataset ini tidak memiliki kolom "churn". Label inaktivitas **dikonstruksi** dari pola transaksi menggunakan definisi temporal yang eksplisit.
