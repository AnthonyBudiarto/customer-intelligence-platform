"""
Halaman 2: Customer Segmentation
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🎯 Customer Segmentation")
st.markdown("Segmentasi pelanggan berdasarkan RFM (Recency, Frequency, Monetary) menggunakan K-Means clustering.")

ROOT = Path(__file__).parents[2]

@st.cache_data
def load_segments():
    seg = pd.read_parquet(ROOT / "data/features/rfm_segments.parquet")
    return seg

@st.cache_data
def load_seg_report():
    path = ROOT / "artifacts/reports/segmentation_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

try:
    seg = load_segments()
    report = load_seg_report()

    best_k = report.get("best_k", 2)
    best_sil = report.get("best_silhouette", 0)
    n_customers = report.get("n_customers_segmented", 0)

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Pelanggan Tersegmentasi", f"{n_customers:,}")
    col2.metric("Jumlah Cluster Optimal", str(best_k))
    col3.metric("Silhouette Score", f"{best_sil:.4f}")

    st.info(f"K=2 dipilih berdasarkan silhouette score tertinggi ({best_sil:.4f}). "
            "Ini mengindikasikan dua kelompok yang well-separated: Champions vs Inactive Low Value.")

    st.markdown("---")

    # Profil segmen
    st.subheader("Profil Segmen")
    profiles = report.get("segment_profiles", [])
    if profiles:
        df_prof = pd.DataFrame(profiles)
        cols_show = ["cluster", "segment_name", "n_customers", "pct_customers",
                     "recency_median", "frequency_median", "monetary_median"]
        cols_available = [c for c in cols_show if c in df_prof.columns]
        df_display = df_prof[cols_available].copy()
        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
        st.dataframe(df_display, use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        # Distribusi pelanggan per segmen
        st.subheader("Distribusi Pelanggan per Segmen")
        if "cluster_label" in seg.columns:
            seg_counts = seg["cluster_label"].value_counts().reset_index()
            seg_counts.columns = ["Segmen", "Jumlah Pelanggan"]
            fig1 = px.pie(seg_counts, values="Jumlah Pelanggan", names="Segmen",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig1.update_layout(height=350)
            st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        # Revenue per segmen
        st.subheader("Revenue per Segmen")
        if "cluster_label" in seg.columns and "monetary_total" in seg.columns:
            seg_rev = seg.groupby("cluster_label")["monetary_total"].sum().reset_index()
            seg_rev.columns = ["Segmen", "Total Revenue"]
            fig2 = px.pie(seg_rev, values="Total Revenue", names="Segmen",
                          color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Distribusi RFM per Segmen")

    col_r1, col_r2, col_r3 = st.columns(3)
    if "cluster_label" in seg.columns:
        with col_r1:
            fig3 = px.box(seg, x="cluster_label", y="recency_days",
                          title="Recency per Segmen",
                          labels={"cluster_label": "Segmen", "recency_days": "Recency (hari)"},
                          color="cluster_label", color_discrete_sequence=px.colors.qualitative.Set2)
            fig3.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig3, use_container_width=True)

        with col_r2:
            fig4 = px.box(seg, x="cluster_label", y=np.log1p(seg["frequency"]),
                          title="Frequency per Segmen (log)",
                          labels={"cluster_label": "Segmen", "y": "log(Frequency)"},
                          color="cluster_label", color_discrete_sequence=px.colors.qualitative.Set2)
            fig4.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig4, use_container_width=True)

        with col_r3:
            fig5 = px.box(seg, x="cluster_label", y=np.log1p(seg["monetary_total"]),
                          title="Monetary per Segmen (log)",
                          labels={"cluster_label": "Segmen", "y": "log(Revenue)"},
                          color="cluster_label", color_discrete_sequence=px.colors.qualitative.Set2)
            fig5.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig5, use_container_width=True)

    # Business actions
    st.markdown("---")
    st.subheader("Rekomendasi Aksi Bisnis per Segmen")
    st.markdown("""
    | Segmen | Karakteristik | Aksi Potensial |
    |--------|--------------|----------------|
    | **Champions** | Beli baru-baru ini, sering, nilai tinggi | Loyalty program, early access produk baru, reward program |
    | **Inactive Low Value** | Sudah lama tidak beli, jarang, nilai rendah | Win-back campaign bertarget, survei alasan inaktivitas |

    > **Catatan:** Rekomendasi ini adalah hipotesis bisnis. Efektivitasnya memerlukan pengujian A/B
    > atau eksperimen terkontrol sebelum dapat diklaim sebagai strategi yang terbukti efektif.
    """)

    # Radar chart image
    st.markdown("---")
    radar_img = ROOT / "artifacts/plots/12_segment_radar.png"
    if radar_img.exists():
        st.image(str(radar_img), caption="Profil RFM per Segmen (dinormalisasi 0-1)")

except Exception as e:
    st.error(f"Gagal memuat segmentasi: {e}")
    st.info("Pastikan run_segmentation.py sudah dijalankan.")
