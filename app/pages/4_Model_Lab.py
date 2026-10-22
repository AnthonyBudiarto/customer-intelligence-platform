"""
Halaman 4: Model Lab — Perbandingan model, curves, calibration
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(page_title="Model Lab", layout="wide")
st.title("🤖 Model Lab")
st.markdown("Perbandingan semua model prediksi inaktivitas pada test set kronologis yang identik.")

ROOT = Path(__file__).parents[2]
PLOTS = ROOT / "artifacts/plots"

@st.cache_data
def load_report():
    path = ROOT / "artifacts/reports/modeling_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

try:
    report = load_report()
    all_metrics = report.get("model_metrics", [])

    if not all_metrics:
        st.warning("Model belum dilatih. Jalankan run_modeling.py.")
        st.stop()

    # Tabel perbandingan
    st.subheader("Perbandingan Semua Model (Test Set)")
    df_m = pd.DataFrame(all_metrics)
    cols = ["model_name", "roc_auc", "pr_auc", "f1", "precision", "recall",
            "balanced_accuracy", "brier_score", "n_flagged", "pct_flagged"]
    df_display = df_m[[c for c in cols if c in df_m.columns]].copy()
    df_display = df_display.rename(columns={
        "model_name": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
        "f1": "F1", "precision": "Precision", "recall": "Recall",
        "balanced_accuracy": "Bal. Acc", "brier_score": "Brier",
        "n_flagged": "N Flagged", "pct_flagged": "% Flagged",
    })

    best_model = report.get("best_model", "")

    def highlight_best(row):
        is_best = row["Model"] == best_model
        return ["background-color: #d4edda" if is_best else "" for _ in row]

    st.dataframe(df_display.style.apply(highlight_best, axis=1), use_container_width=True)
    st.success(f"Model terbaik (PR-AUC): **{best_model}**")

    st.markdown("""
    **Cara membaca:**
    - **PR-AUC**: Metrik utama — lebih relevan dari ROC-AUC ketika kelas tidak seimbang.
    - **Brier Score**: Kalibrasi probabilitas — semakin rendah semakin baik.
    - **% Flagged**: Proporsi pelanggan yang ditandai sebagai berisiko pada threshold terpilih.
    """)

    # Business context per model
    st.subheader("Interpretasi Bisnis per Model")
    selected_model = st.selectbox("Pilih model:", [m["model_name"] for m in all_metrics])
    sel = next(m for m in all_metrics if m["model_name"] == selected_model)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TP (Benar Inactive)", str(sel.get("tp", "N/A")))
    col2.metric("FP (Salah Flagged)", str(sel.get("fp", "N/A")))
    col3.metric("FN (Missed Inactive)", str(sel.get("fn", "N/A")))
    col4.metric("TN (Benar Active)", str(sel.get("tn", "N/A")))

    flagged = sel.get("n_flagged", 0)
    total = sel.get("n_samples", 1)
    prec = sel.get("precision", 0)
    rec = sel.get("recall", 0)

    st.markdown(f"""
    Pada threshold **{sel.get('threshold', 0.5):.3f}**:
    - **{flagged}** pelanggan ({sel.get('pct_flagged', 0):.1f}%) ditandai berisiko tinggi
    - **{prec*100:.1f}%** dari yang ditandai benar-benar inactive (Precision)
    - **{rec*100:.1f}%** dari semua inactive berhasil teridentifikasi (Recall)

    > **False Positive** (salah flagged) berarti tim retensi membuang sumber daya pada pelanggan yang sebenarnya masih aktif.
    > **False Negative** (terlewat) berarti pelanggan yang akan churn tidak mendapat perhatian.
    """)

    st.markdown("---")

    # Plot images
    plot_tab1, plot_tab2, plot_tab3 = st.tabs(["PR Curves", "ROC Curves", "Calibration"])

    with plot_tab1:
        pr_img = PLOTS / "21_pr_curves.png"
        if pr_img.exists():
            st.image(str(pr_img))
        else:
            st.info("Plot belum tersedia.")

    with plot_tab2:
        roc_img = PLOTS / "22_roc_curves.png"
        if roc_img.exists():
            st.image(str(roc_img))

    with plot_tab3:
        cal_img = PLOTS / "23_calibration.png"
        if cal_img.exists():
            st.image(str(cal_img))
        st.caption("Garis putus-putus diagonal = perfect calibration. Semakin dekat ke garis, "
                   "semakin dapat dipercaya probabilitasnya.")

    st.markdown("---")

    # SHAP
    st.subheader("Explainability — SHAP (XGBoost)")
    shap_img = PLOTS / "24_shap_global_importance.png"
    shap_sum = PLOTS / "25_shap_summary.png"

    if shap_img.exists():
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.image(str(shap_img), caption="Global Feature Importance (Mean |SHAP|)")
        with col_s2:
            if shap_sum.exists():
                st.image(str(shap_sum), caption="SHAP Summary Plot")

        shap_top = report.get("shap_top_features", [])
        if shap_top:
            st.markdown(f"""
            **Top 5 fitur berdasarkan SHAP:** {', '.join(shap_top)}

            > SHAP mendeskripsikan **kontribusi fitur terhadap prediksi model**, bukan kausalitas.
            > Fitur penting dalam model bisa berkorelasi dengan inaktivitas tanpa menyebabkannya.
            """)

    # Confusion matrices
    st.markdown("---")
    st.subheader("Confusion Matrices")
    cm_models = {
        "Rule-Based": "cm_rule-based.png",
        "Logistic Regression": "cm_logistic_regression.png",
        "Random Forest": "cm_random_forest.png",
        "XGBoost": "cm_xgboost.png",
        "MLP": "cm_mlp.png",
    }
    cols = st.columns(len(cm_models))
    for i, (name, fname) in enumerate(cm_models.items()):
        img_path = PLOTS / fname
        if img_path.exists():
            cols[i].image(str(img_path), caption=name, use_column_width=True)

    # MLP training curve
    st.markdown("---")
    mlp_curve = PLOTS / "20_mlp_training_curve.png"
    if mlp_curve.exists():
        st.subheader("MLP Training Curve")
        st.image(str(mlp_curve))

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Jalankan run_modeling.py terlebih dahulu.")
