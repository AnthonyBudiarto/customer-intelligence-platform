"""
AI-Powered Customer Intelligence & Retention Platform
Dashboard Utama — Streamlit Multi-Page App
"""
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="favicon.ico" if Path("favicon.ico").exists() else ":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS tambahan
st.markdown("""
<style>
.big-metric { font-size: 2rem; font-weight: bold; color: #1f77b4; }
.metric-label { font-size: 0.85rem; color: #666; }
.warning-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; border-radius: 4px; margin: 8px 0; }
.info-box { background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; border-radius: 4px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered Customer Intelligence & Retention Platform")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Tentang Platform Ini

    Platform ini membangun sistem prediksi inaktivitas pelanggan berbasis data historis transaksi
    dari retailer online UK (2010–2011). Seluruh analisis menggunakan data aktual yang diaudit
    dan divalidasi secara metodologis.

    **Navigasi menggunakan sidebar kiri:**
    """)
    st.markdown("""
    | Halaman | Konten |
    |---------|--------|
    | 📊 Customer Analytics | Revenue, frekuensi, recency, monetary trends |
    | 🎯 Segmentation | RFM clustering, profil segmen, business actions |
    | ⚠️ Retention Risk | Risk scores, ranked list, prioritization |
    | 🤖 Model Lab | Perbandingan model, PR/ROC curves, kalibrasi |
    | 📖 Methodology | Desain temporal, definisi, limitasi |
    """)

with col2:
    st.markdown("### KPI Dataset")
    st.metric("Total Transaksi Raw", "541,909")
    st.metric("Pelanggan Teridentifikasi", "4,372")
    st.metric("Periode Data", "373 hari")
    st.metric("Total Gross Revenue", "£8.9 Juta")
    st.metric("Pelanggan Test Set", "2,124")

st.markdown("---")
st.markdown("""
<div class='warning-box'>
<b>Catatan Penting:</b> Platform ini menggunakan data historis 2010-2011 dari UCI Machine Learning Repository.
Prediksi bersifat estimasi statistik — bukan jaminan bahwa pelanggan akan atau tidak akan kembali membeli.
Intervensi bisnis (diskon, kampanye) memerlukan evaluasi kausal terpisah.
</div>
""", unsafe_allow_html=True)
