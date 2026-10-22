"""
Fix SHAP explanations untuk XGBoost (versi baru memerlukan cara berbeda).
"""
import sys, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.models.preprocessing import fit_transform_train, transform_test, FEATURE_COLUMNS

PLOTS_DIR = Path("artifacts/plots")
REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
FEATURES_DIR = Path("data/features")

print("Memuat data dan model...")
X_train = pd.read_parquet(FEATURES_DIR / "features_train.parquet")
y_train = pd.read_parquet(FEATURES_DIR / "target_train.parquet")["inactive"]
X_test  = pd.read_parquet(FEATURES_DIR / "features_test.parquet")
y_test  = pd.read_parquet(FEATURES_DIR / "target_test.parquet")["inactive"]

pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
X_train_np, _ = fit_transform_train(X_train)
X_test_np = transform_test(X_test, pipeline)

xgb_model = joblib.load(MODELS_DIR / "xgboost.joblib")
feature_names = FEATURE_COLUMNS

import shap
print(f"SHAP version: {shap.__version__}")

# Gunakan Explainer generik (tidak spesifik TreeExplainer)
try:
    # Coba pendekatan baru
    explainer = shap.Explainer(xgb_model, X_train_np,
                               feature_names=feature_names)
    shap_values = explainer(X_test_np)

    # Untuk XGBoost binary, shap_values.values sudah (n, n_features)
    if hasattr(shap_values, "values"):
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # class 1 (inactive)
    else:
        sv = shap_values

    shap_mean = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_mean,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\nTop 10 fitur (SHAP):")
    print(shap_df.head(10).to_string(index=False))

    # Bar plot
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1],
            color="steelblue", alpha=0.8)
    ax.set_title("Global Feature Importance (Mean |SHAP|)\nXGBoost Model", fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "24_shap_global_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # SHAP dot summary
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test_np, feature_names=feature_names, show=False, plot_type="dot")
    plt.title("SHAP Summary Plot — XGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "25_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    shap_df.to_csv(str(REPORTS_DIR / "shap_feature_importance.csv"), index=False)
    np.save(str(MODELS_DIR / "shap_values_test.npy"), sv)

    # Update modeling report
    report_path = REPORTS_DIR / "modeling_report.json"
    with open(report_path) as f:
        report = json.load(f)
    report["shap_ok"] = True
    report["shap_top_features"] = shap_df.head(5)["feature"].tolist()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("SHAP selesai!")

except Exception as e:
    print(f"SHAP Explainer gagal: {e}")

    # Fallback: gunakan feature importance dari XGBoost langsung
    print("Menggunakan XGBoost built-in feature importance sebagai fallback...")
    fi = xgb_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
    print(fi_df.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color="steelblue", alpha=0.8)
    ax.set_title("Feature Importance (XGBoost gain)\nFallback dari SHAP", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "24_shap_global_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    fi_df.to_csv(str(REPORTS_DIR / "shap_feature_importance.csv"), index=False)

    report_path = REPORTS_DIR / "modeling_report.json"
    with open(report_path) as f:
        report = json.load(f)
    report["shap_ok"] = False
    report["shap_note"] = "Menggunakan XGBoost built-in importance sebagai fallback"
    report["shap_top_features"] = fi_df.head(5)["feature"].tolist()
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("Feature importance tersimpan (fallback mode).")
