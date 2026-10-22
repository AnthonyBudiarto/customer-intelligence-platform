"""
Utilitas untuk memuat konfigurasi proyek dari YAML.
"""

from pathlib import Path
import yaml
from typing import Any


def load_config(config_path: str = None) -> dict:
    """
    Muat konfigurasi proyek dari file YAML.

    Parameters
    ----------
    config_path : str, optional
        Path ke file konfigurasi. Default: configs/project_config.yaml
        relatif terhadap root proyek.

    Returns
    -------
    dict
        Dictionary berisi semua parameter konfigurasi.
    """
    if config_path is None:
        # Cari root proyek dari lokasi file ini (src/utils/config.py → root)
        root = Path(__file__).resolve().parents[2]
        config_path = root / "configs" / "project_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {config_path}\n"
            f"Pastikan configs/project_config.yaml ada di root proyek."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Return absolute path ke root proyek."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """
    Ubah path relatif (dari config) menjadi path absolut.

    Parameters
    ----------
    relative_path : str
        Path relatif terhadap root proyek, misal "data/raw/Online Retail.xlsx"

    Returns
    -------
    Path
        Path absolut yang sudah divalidasi keberadaannya.
    """
    return get_project_root() / relative_path
