"""
Halaman 3: Retention Risk — Risk scores dan ranked customer list
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Retention Risk", layout="wide")
st.title("⚠️ Retention Risk")
st.markdown("Pelanggan diranking berdasarkan kombinasi inactivity risk score dan nilai historis.")

ROOT = Path(__file__).parents[2]

@st.cache_data
def load_retention():
    path = ROOT / "data/features/retention_priority_list.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df.index.name = "CustomerID"
        return df
    return None

@st.cache_data
def load_modeling_report():
    path = ROOT / "artifacts/reports/modeling_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

try:
    df = load_retention()
    report = load_modeling_report()

    if df is None:
        st.warning("Data retention belum tersedia. Jalankan run_modeling.py terlebih dahulu.")
        st.stop()

    # KPIs
    ret_summary = report.get("retention_summary", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pelanggan", f"{len(df):,}")
    col2.metric("High Risk", f"{ret_summary.get('n_high_risk', 0):,}")
    col3.metric("Medium Risk", f"{ret_summary.get('n_medium_risk', 0):,}")
    col4.metric("Low Risk", f"{ret_summary.get('n_low_risk', 0):,}")

    st.markdown("""
    <div style='background:#fff3cd;border-left:4px solid #ffc107;padding:10px;border-radius:4px;margin:8px 0'>
    <b>Formula Prioritization:</b><br>
    Priority Score = 0.6 × Inactivity Risk + 0.4 × Customer Value Score<br>
    Risk tiers: Low (&lt;33%), Medium (33-67%), High (&gt;67%)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk distribution
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Distribusi Risk Score")
        fig1 = px.histogram(df, x="inactivity_prob", nbins=40,
                            title="Distribusi Inactivity Risk Score (XGBoost)",
                            labels={"inactivity_prob": "Risk Score", "count": "Pelanggan"},
                            color_discrete_sequence=["#e74c3c"])
        xgb_threshold = report.get("thresholds", {}).get("xgboost", 0.5)
        fig1.add_vline(x=xgb_threshold, line_dash="dash", line_color="black",
                       annotation_text=f"Threshold={xgb_threshold:.2f}")
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

    with col_r:
        st.subheader("Prioritization Matrix: Risk vs Value")
        scatter_color = df["true_label"].map({0: "Active", 1: "Inactive"}) if "true_label" in df.columns else None
        if scatter_color is not None:
            fig2 = px.scatter(df, x="inactivity_prob",
                              y=np.log1p(df["customer_revenue"]),
                              color=scatter_color,
                              color_discrete_map={"Active": "green", "Inactive": "red"},
                              title="Risk vs Value (warna = label aktual)",
                              labels={"inactivity_prob": "Risk Score",
                                      "y": "log(Revenue)",
                                      "color": "Label Aktual"},
                              opacity=0.5)
        else:
            fig2 = px.scatter(df, x="inactivity_prob",
                              y=np.log1p(df["customer_revenue"]),
                              title="Risk vs Value")
        fig2.add_vline(x=xgb_threshold, line_dash="dash", line_color="black")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Daftar Pelanggan Berdasarkan Prioritas Retensi")

    # Filter
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        risk_filter = st.multiselect("Filter Risk Tier",
                                     options=["High Risk", "Medium Risk", "Low Risk"],
                                     default=["High Risk", "Medium Risk"])
    with col_f2:
        top_n = st.slider("Tampilkan Top N Pelanggan", 10, 200, 50)
    with col_f3:
        show_cols = st.multiselect("Kolom Ditampilkan",
                                   options=[c for c in df.columns],
                                   default=["inactivity_prob", "customer_revenue",
                                            "priority_score", "risk_tier",
                                            "recency_days", "frequency"])

    if "risk_tier" in df.columns and risk_filter:
        filtered = df[df["risk_tier"].isin(risk_filter)].sort_values("priority_rank")
    else:
        filtered = df.sort_values("priority_rank")

    display_df = filtered.head(top_n)[[c for c in show_cols if c in filtered.columns]].copy()

    # Format columns
    if "inactivity_prob" in display_df.columns:
        display_df["inactivity_prob"] = display_df["inactivity_prob"].map("{:.3f}".format)
    if "customer_revenue" in display_df.columns:
        display_df["customer_revenue"] = display_df["customer_revenue"].map("£{:,.0f}".format)
    if "priority_score" in display_df.columns:
        display_df["priority_score"] = display_df["priority_score"].map("{:.3f}".format)

    st.dataframe(display_df, use_container_width=True, height=400)
    st.caption(f"Menampilkan {len(display_df):,} dari {len(filtered):,} pelanggan yang difilter.")

    st.markdown("---")
    st.markdown("""
    ### Cara Membaca Tabel Ini

    - **inactivity_prob**: Probabilitas 0–1 dari model XGBoost bahwa pelanggan tidak akan membeli dalam 70 hari ke depan.
    - **customer_revenue**: Total revenue historis pelanggan (dari dataset observasi).
    - **priority_score**: Skor gabungan (60% risk + 40% value) untuk menentukan urutan prioritas tim retensi.
    - **risk_tier**: Kategori risiko berdasarkan inactivity_prob.

    > **Disclaimer**: Ini adalah estimasi statistik berdasarkan pola historis. Pelanggan high-risk
    > yang sebenarnya aktif tidak selalu perlu diintervensi. Validasi dengan konteks bisnis tambahan.
    """)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Jalankan run_modeling.py terlebih dahulu.")
