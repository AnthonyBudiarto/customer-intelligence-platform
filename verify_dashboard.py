"""Verifikasi semua file yang dibutuhkan dashboard tersedia."""
from pathlib import Path
import sys

ROOT = Path(__file__).parent
checks = [
    # Data
    "data/processed/cleaned_modeling.parquet",
    "data/processed/cleaned_analysis.parquet",
    "data/features/features_train.parquet",
    "data/features/features_test.parquet",
    "data/features/target_train.parquet",
    "data/features/target_test.parquet",
    "data/features/rfm_segments.parquet",
    "data/features/retention_priority_list.parquet",
    # Models
    "artifacts/models/preprocessing_pipeline.joblib",
    "artifacts/models/xgboost.joblib",
    "artifacts/models/logistic_regression.joblib",
    "artifacts/models/random_forest.joblib",
    "artifacts/models/mlp_model.pt",
    "artifacts/models/kmeans_model.joblib",
    # Reports
    "artifacts/reports/audit_report.json",
    "artifacts/reports/eda_stats.json",
    "artifacts/reports/segmentation_report.json",
    "artifacts/reports/modeling_report.json",
    # Key plots
    "artifacts/plots/21_pr_curves.png",
    "artifacts/plots/22_roc_curves.png",
    "artifacts/plots/24_shap_global_importance.png",
    # App files
    "app/main.py",
    "app/pages/1_Customer_Analytics.py",
    "app/pages/2_Segmentation.py",
    "app/pages/3_Retention_Risk.py",
    "app/pages/4_Model_Lab.py",
    "app/pages/5_Methodology.py",
    # Config
    "configs/project_config.yaml",
    "README.md",
]

all_ok = True
for f in checks:
    path = ROOT / f
    status = "OK" if path.exists() else "MISSING"
    if status == "MISSING":
        all_ok = False
    print(f"  [{status}] {f}")

print()
if all_ok:
    print("SEMUA FILE TERSEDIA — Dashboard siap dijalankan.")
    print("Jalankan: streamlit run app/main.py")
else:
    print("Ada file yang hilang — periksa di atas.")
    sys.exit(1)
