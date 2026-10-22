"""
Konfigurasi logging terpusat menggunakan loguru.
Semua modul src harus mengimpor logger dari sini.
"""

import sys
from pathlib import Path
from loguru import logger

# Hapus handler default
logger.remove()

# Handler ke console dengan format ringkas
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Handler ke file log
log_dir = Path(__file__).resolve().parents[2] / "logs"
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "project.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)


def get_logger(name: str = None):
    """Return logger instance dengan nama modul opsional."""
    if name:
        return logger.bind(name=name)
    return logger
