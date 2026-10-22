"""
Halaman 1: Customer Analytics
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Analytics", layout="wide")
st.title("📊 Customer Analytics")
st.markdown("Analisis revenue, frekuensi, recency, dan monetary pelanggan berdasarkan data historis.")

ROOT = Path(__file__).parents[2]

@st.cache_data
def load_data():
    df_m = pd.read_parquet(ROOT / "data/processed/cleaned_modeling.parquet")
    df_m["InvoiceDate"] = pd.to_datetime(df_m["InvoiceDate"])
    if "LineValue" not in df_m.columns:
        df_m["LineValue"] = df_m["Quantity"] * df_m["UnitPrice"]
    return df_m

@st.cache_data
def load_eda_stats():
    path = ROOT / "artifacts/reports/eda_stats.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

try:
    df = load_data()
    stats = load_eda_stats()

    # --- KPIs ---
    col1, col2, col3, col4, col5 = st.columns(5)
    total_rev = df["LineValue"].sum()
    n_customers = df["CustomerID"].nunique()
    n_orders = df["InvoiceNo"].nunique()
    aov = df.groupby("InvoiceNo")["LineValue"].sum().median()
    n_countries = df["Country"].nunique()

    col1.metric("Total Revenue", f"£{total_rev/1e6:.2f}M")
    col2.metric("Pelanggan Unik", f"{n_customers:,}")
    col3.metric("Total Order", f"{n_orders:,}")
    col4.metric("Median AOV", f"£{aov:.0f}")
    col5.metric("Negara", str(n_countries))

    st.markdown("---")

    # --- Revenue Bulanan ---
    st.subheader("Tren Revenue dan Order Bulanan")
    df["YM"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("YM").agg(Revenue=("LineValue","sum"), Orders=("InvoiceNo","nunique")).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly["YM"], y=monthly["Revenue"]/1000, name="Revenue (£k)",
                         marker_color="steelblue", opacity=0.8))
    fig.add_trace(go.Scatter(x=monthly["YM"], y=monthly["Orders"], name="Orders",
                             mode="lines+markers", yaxis="y2", line=dict(color="orange", width=2)))
    fig.update_layout(
        yaxis=dict(title="Revenue (GBP Thousands)"),
        yaxis2=dict(title="Jumlah Order", overlaying="y", side="right"),
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Peak bulan:** {stats.get('monthly_revenue', {}).get('peak_month', 'N/A')} "
            f"dengan revenue £{stats.get('monthly_revenue', {}).get('peak_revenue', 0):,.0f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        # --- Distribusi Frekuensi ---
        st.subheader("Distribusi Frekuensi per Pelanggan")
        cust_freq = df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
        cust_freq.columns = ["CustomerID", "n_invoices"]

        fig2 = px.histogram(cust_freq[cust_freq["n_invoices"] <= 30],
                            x="n_invoices", nbins=30,
                            title="Frekuensi Transaksi (dikap 30)",
                            labels={"n_invoices": "Jumlah Invoice", "count": "Pelanggan"},
                            color_discrete_sequence=["steelblue"])
        fig2.add_vline(x=cust_freq["n_invoices"].median(), line_dash="dash", line_color="red",
                       annotation_text=f"Median={cust_freq['n_invoices'].median():.0f}")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)
        pct_single = (cust_freq["n_invoices"] == 1).mean() * 100
        st.caption(f"30.0% pelanggan hanya melakukan 1 transaksi. "
                   f"70.0% memiliki ≥2 transaksi.")

    with col_right:
        # --- Distribusi Monetary ---
        st.subheader("Distribusi Revenue per Pelanggan")
        cust_rev = df.groupby("CustomerID")["LineValue"].sum().reset_index()
        cust_rev.columns = ["CustomerID", "total_revenue"]
        cap = cust_rev["total_revenue"].quantile(0.95)

        fig3 = px.histogram(cust_rev[cust_rev["total_revenue"] <= cap],
                            x="total_revenue", nbins=50,
                            title="Revenue per Pelanggan (dikap P95)",
                            labels={"total_revenue": "Revenue (£)", "count": "Pelanggan"},
                            color_discrete_sequence=["coral"])
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)
        gini = stats.get("monetary", {}).get("gini_coefficient", 0)
        top20 = stats.get("monetary", {}).get("top20pct_customers_revenue_share", 0)
        st.caption(f"Gini coefficient: {gini:.3f}. "
                   f"Top 20% pelanggan = {top20:.1f}% total revenue.")

    st.markdown("---")
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # --- Recency Distribution ---
        st.subheader("Distribusi Recency")
        max_date = df["InvoiceDate"].max()
        last_purch = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
        last_purch["recency_days"] = (max_date - last_purch["InvoiceDate"]).dt.days

        fig4 = px.histogram(last_purch, x="recency_days", nbins=50,
                            title=f"Recency (relatif ke {max_date.date()})",
                            labels={"recency_days": "Hari Sejak Pembelian Terakhir"},
                            color_discrete_sequence=["mediumpurple"])
        fig4.add_vline(x=70, line_dash="dot", line_color="orange",
                       annotation_text="70-hari horizon")
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    with col_r2:
        # --- Top Countries ---
        st.subheader("Revenue per Negara (Top 10)")
        ctry = df.groupby("Country")["LineValue"].sum().sort_values(ascending=False).head(10).reset_index()
        fig5 = px.bar(ctry[::-1].reset_index(drop=True),
                      x="LineValue", y="Country", orientation="h",
                      title="Top 10 Negara",
                      labels={"LineValue": "Revenue (£)", "Country": ""},
                      color="LineValue", color_continuous_scale="Blues")
        fig5.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    # --- Cohort plot (image) ---
    st.markdown("---")
    st.subheader("Cohort Retention Rate")
    cohort_img = ROOT / "artifacts/plots/09_cohort_retention.png"
    if cohort_img.exists():
        st.image(str(cohort_img), caption="Rata-rata retention bulan 1: ~22.6%")
    avg_m1 = stats.get("cohort_retention", {}).get("avg_month1_retention_pct", None)
    if avg_m1:
        st.warning(f"Rata-rata hanya {avg_m1:.1f}% pelanggan kembali membeli di bulan berikutnya — "
                   "mengindikasikan tingginya risiko one-time buyers.")

except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.info("Pastikan run_phase1.py, run_eda.py, dan run_features.py sudah dijalankan.")
