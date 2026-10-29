# AI-Powered Customer Intelligence & Retention Platform

Proyek Data Science end-to-end untuk prediksi inaktivitas pelanggan, segmentasi RFM, dan prioritization retensi menggunakan data transaksi historis.

---

## Hasil Utama

| Model | ROC-AUC | PR-AUC | F1 |
|-------|---------|--------|----|
| Rule-Based (Recency) | 0.557 | 0.427 | 0.504 |
| Logistic Regression | 0.728 | 0.561 | 0.614 |
| Random Forest | 0.721 | 0.552 | 0.602 |
| XGBoost | 0.697 | 0.527 | 0.512 |
| **MLP (PyTorch)** | **0.733** | **0.569** | **0.620** |

**Kesimpulan:** ML model memberikan peningkatan bermakna dibanding rule-based baseline (+14% PR-AUC). MLP dan Logistic Regression hampir setara, kesederhanaan Logistic Regression menjadi argumen kuat untuk deployment.

**Top fitur prediktif:** n_unique_months (42%), frequency (24%), monetary_total (6%)

---

## Dataset

- **Sumber:** [UCI Online Retail Dataset](https://archive.uci.edu/dataset/352/online+retail)
- **Periode:** 01 Desember 2010 – 09 Desember 2011
- **Raw:** 541,909 transaksi | 4,372 pelanggan teridentifikasi
- **Tidak diikutsertakan dalam repo**, lihat `data/raw/README.md` untuk instruksi unduh

---

## Instalasi

### 1. Clone repository

```bash
git clone <repo-url>
cd customer-intelligence-platform
```

### 2. Buat conda environment

```bash
conda env create -f environment.yml
conda activate cip-env
```

### 3. Unduh dataset

Lihat instruksi di `data/raw/README.md`. Simpan file sebagai:
```
data/raw/Online Retail.xlsx
```

---

## Cara Menjalankan

Semua script dijalankan dari root proyek dengan conda environment aktif.

```bash
# Fase 1: Data Audit & Cleaning
python run_phase1.py

# Fase 2: EDA & Temporal Validation
python run_eda.py

# Fase 3: Feature Engineering & Target
python run_features.py

# Fase 4: Segmentasi
python run_segmentation.py

# Fase 5+6: Semua Model + Evaluasi + Retention
python run_modeling.py

# Fix SHAP (jika diperlukan)
python run_shap_fix.py

# Jalankan tests
pytest tests/ -v

# Dashboard
streamlit run app/main.py
```

---

## Struktur Repository

```
customer-intelligence-platform/
├── README.md
├── environment.yml          # Conda environment
├── configs/
│   └── project_config.yaml  # Semua parameter temporal & model
├── data/
│   ├── raw/                 # Dataset (tidak di-commit)
│   ├── processed/           # Cleaned data (parquet)
│   └── features/            # Features, targets, segments
├── src/
│   ├── data/                # loader, auditor, cleaner
│   ├── features/            # feature_engineering, target_construction
│   ├── segmentation/        # rfm_segmentation
│   ├── models/              # preprocessing
│   └── evaluation/          # metrics
├── tests/                   # Unit tests temporal integrity
├── app/
│   ├── main.py              # Dashboard homepage
│   └── pages/               # 5 halaman dashboard
├── artifacts/
│   ├── models/              # Trained models (joblib/pt)
│   ├── plots/               # Semua visualisasi
│   └── reports/             # JSON reports
└── reports/                 # Markdown reports
```

---

## Desain Temporal

```
Dataset:    01 Des 2010 ─────────────────────────── 09 Des 2011

Train:      ─── obs window ──── | cutoff 30 Jun 2011 | ── 70d horizon ──|

Test:       ─── obs window ─────────── | cutoff 30 Sep 2011 | ── 70d ──|
```

**Tidak ada data leakage:** Fitur hanya menggunakan transaksi pada atau sebelum cutoff. Preprocessing (scaler, imputer) difit hanya pada training data.

---

## Segmentasi

K-Means dengan k=2 (silhouette=0.430):

| Segmen | % Pelanggan | Recency Median | Freq Median | Monetary Median |
|--------|-------------|----------------|-------------|-----------------|
| Champions | 36% | 17 hari | 5 tx | £1,853 |
| Inactive Low Value | 64% | 116 hari | 1 tx | £336 |

---

## Limitasi Penting

- Data 2010-2011: tidak representatif untuk e-commerce modern
- Satu snapshot temporal: stabilitas model tidak terukur
- 25% transaksi tidak memiliki CustomerID (bias seleksi)
- "Inaktivitas" ≠ churn: pelanggan mungkin membeli di saluran lain
- Prediksi tidak kausal: intervensi memerlukan eksperimen terkontrol

---

## Teknologi

Python 3.10 | pandas | scikit-learn | XGBoost | PyTorch | SHAP | Streamlit | Plotly

---

## Sitasi Dataset

Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry.
*Journal of Database Marketing & Customer Strategy Management*, 19(3), 197-208.
