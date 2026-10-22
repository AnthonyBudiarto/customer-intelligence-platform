"""
Unit tests untuk temporal integrity — memastikan tidak ada data leakage.
Jalankan dengan: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def df_modeling():
    path = ROOT / "data/processed/cleaned_modeling.parquet"
    if not path.exists():
        pytest.skip("cleaned_modeling.parquet belum ada — jalankan run_phase1.py")
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if "LineValue" not in df.columns:
        df["LineValue"] = df["Quantity"] * df["UnitPrice"]
    return df


@pytest.fixture
def df_analysis():
    path = ROOT / "data/processed/cleaned_analysis.parquet"
    if not path.exists():
        pytest.skip("cleaned_analysis.parquet belum ada")
    df = pd.read_parquet(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if "IsCancelled" not in df.columns:
        df["IsCancelled"] = df["InvoiceNo"].str.startswith("C", na=False)
    return df


class TestTargetConstruction:
    """Tests untuk konstruksi target inaktivitas."""

    def test_target_no_future_data(self, df_modeling):
        """Target harus dibangun dari data SETELAH cutoff — tidak sebelum."""
        from src.features.target_construction import construct_inactivity_target

        cutoff = pd.Timestamp("2011-09-30")
        horizon_days = 70
        labels, stats = construct_inactivity_target(
            df_modeling, cutoff, horizon_days
        )

        # Label tidak boleh berisi nilai selain 0 dan 1
        assert set(labels.unique()).issubset({0, 1}), "Label harus 0 atau 1"

        # Semua pelanggan berlabel harus pernah beli sebelum cutoff
        pre_cutoff_customers = set(
            df_modeling[df_modeling["InvoiceDate"] <= cutoff]["CustomerID"].unique()
        )
        for cid in labels.index:
            assert cid in pre_cutoff_customers, \
                f"CustomerID {cid} tidak memiliki transaksi sebelum cutoff!"

    def test_horizon_not_exceed_dataset(self, df_modeling):
        """Horizon tidak boleh melampaui akhir dataset."""
        from src.features.target_construction import construct_inactivity_target

        cutoff = pd.Timestamp("2011-09-30")
        dataset_end = df_modeling["InvoiceDate"].max()

        # Horizon yang terlalu panjang harus raise ValueError
        too_long = (dataset_end - cutoff).days + 10
        with pytest.raises(ValueError, match="melampaui akhir dataset"):
            construct_inactivity_target(df_modeling, cutoff, horizon_days=too_long)

    def test_class_distribution_reasonable(self, df_modeling):
        """Distribusi kelas harus reasonable (tidak all-0 atau all-1)."""
        from src.features.target_construction import construct_inactivity_target

        cutoff = pd.Timestamp("2011-09-30")
        labels, stats = construct_inactivity_target(df_modeling, cutoff, 70)

        pct_inactive = stats["pct_inactive"]
        assert 5 <= pct_inactive <= 95, \
            f"Distribusi kelas ekstrem: {pct_inactive:.1f}% inactive. Cek definisi eligibility."


class TestFeatureEngineering:
    """Tests untuk feature engineering — no leakage."""

    def test_recency_non_negative(self, df_modeling, df_analysis):
        """recency_days tidak boleh negatif (ini berarti ada data masa depan)."""
        from src.features.feature_engineering import build_customer_features

        cutoff = pd.Timestamp("2011-09-30")
        features = build_customer_features(df_modeling, df_analysis, cutoff)

        assert (features["recency_days"] >= 0).all(), \
            "recency_days negatif! Ada data setelah cutoff yang masuk ke fitur."

    def test_no_future_transactions_in_features(self, df_modeling, df_analysis):
        """Fitur hanya boleh menggunakan transaksi <= cutoff."""
        from src.features.feature_engineering import build_customer_features

        cutoff = pd.Timestamp("2011-06-30")
        features = build_customer_features(df_modeling, df_analysis, cutoff)

        # Verifikasi: last purchase untuk setiap pelanggan dalam train set
        # harus tidak melebihi cutoff + recency_days (konsistensi internal)
        test_customers = features.index[:20]  # sample 20
        pre_obs = df_modeling[
            (df_modeling["CustomerID"].isin(test_customers))
            & (df_modeling["InvoiceDate"] <= cutoff)
        ]

        for cid in test_customers:
            cust_data = pre_obs[pre_obs["CustomerID"] == cid]
            if len(cust_data) > 0:
                last_tx = cust_data["InvoiceDate"].max()
                expected_recency = (cutoff - last_tx).days
                actual_recency = features.loc[cid, "recency_days"]
                assert abs(expected_recency - actual_recency) <= 1, \
                    f"CustomerID {cid}: expected recency {expected_recency}, got {actual_recency}"

    def test_frequency_positive(self, df_modeling, df_analysis):
        """Frequency harus >= 1 untuk semua pelanggan yang diikutsertakan."""
        from src.features.feature_engineering import build_customer_features

        cutoff = pd.Timestamp("2011-09-30")
        features = build_customer_features(df_modeling, df_analysis, cutoff)

        assert (features["frequency"] >= 1).all(), \
            "Ada pelanggan dengan frequency = 0!"

    def test_monetary_positive(self, df_modeling, df_analysis):
        """monetary_total harus positif."""
        from src.features.feature_engineering import build_customer_features

        cutoff = pd.Timestamp("2011-09-30")
        features = build_customer_features(df_modeling, df_analysis, cutoff)

        assert (features["monetary_total"] > 0).all(), \
            "Ada pelanggan dengan monetary_total <= 0!"


class TestDataCleaning:
    """Tests untuk cleaning pipeline."""

    def test_no_cancelled_in_modeling(self, df_modeling):
        """Dataset modeling tidak boleh mengandung invoice cancelled."""
        cancelled = df_modeling["InvoiceNo"].str.startswith("C", na=False)
        assert cancelled.sum() == 0, \
            f"{cancelled.sum()} cancelled invoices ditemukan dalam cleaned_modeling.parquet!"

    def test_no_negative_quantity_in_modeling(self, df_modeling):
        """Dataset modeling tidak boleh memiliki quantity negatif."""
        assert (df_modeling["Quantity"] >= 1).all(), \
            "Ada quantity < 1 dalam cleaned_modeling.parquet!"

    def test_no_null_customer_id(self, df_modeling):
        """Semua baris dalam dataset modeling harus memiliki CustomerID."""
        assert df_modeling["CustomerID"].notnull().all(), \
            "Ada CustomerID null dalam cleaned_modeling.parquet!"

    def test_no_null_customer_id_analysis(self, df_analysis):
        """Dataset analysis juga tidak boleh memiliki CustomerID null."""
        assert df_analysis["CustomerID"].notnull().all(), \
            "Ada CustomerID null dalam cleaned_analysis.parquet!"

    def test_positive_unit_price_in_modeling(self, df_modeling):
        """UnitPrice harus >= 0.01 dalam dataset modeling."""
        assert (df_modeling["UnitPrice"] >= 0.01).all(), \
            "Ada UnitPrice < 0.01 dalam cleaned_modeling.parquet!"
