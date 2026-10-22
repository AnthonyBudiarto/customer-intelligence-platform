# Data README — Online Retail Dataset

## Cara Mendapatkan Dataset

Dataset ini **tidak diikutsertakan dalam repository** karena berisi data transaksi.

### Unduh dari Sumber Resmi

1. Kunjungi: https://archive.uci.edu/dataset/352/online+retail
2. Klik tombol "Download" untuk mendapatkan file `online_retail.zip`
3. Ekstrak dan simpan file Excel sebagai:
   ```
   data/raw/Online Retail.xlsx
   ```

### Verifikasi Integritas

Setelah mengunduh, jalankan:
```bash
python run_phase1.py
```

Output yang diharapkan:
- Dataset dimuat: **541,909 baris × 8 kolom**
- Rentang tanggal: **2010-12-01 hingga 2011-12-09**
- Ukuran file: ~22.6 MB

### Metadata Dataset

| Atribut | Nilai |
|---------|-------|
| Sumber | UCI ML Repository |
| Donator | Dr Daqing Chen, London South Bank University |
| Lisensi | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Tahun | 2015 |
| Baris | 541,909 |
| Kolom | 8 |
| Rentang | 01 Des 2010 – 09 Des 2011 |
| Atribusi | Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. Journal of Database Marketing & Customer Strategy Management, 19(3), 197-208. |

### Sitasi

```
Dua Daqing Chen, Sain, S.L., and Guo, K. (2012), Data mining for the online 
retail industry: A case study of RFM model-based customer segmentation using 
data mining, Journal of Database Marketing and Customer Strategy Management, 
Vol. 19, No. 3, pp. 197-208.
```
