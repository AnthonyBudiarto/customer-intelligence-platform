"""
Modul evaluasi model — metrics lengkap dan perbandingan.

Setiap fungsi menerima y_true, y_pred, y_prob dan mengembalikan dict terstruktur.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix,
    precision_recall_curve, roc_curve,
    brier_score_loss, classification_report,
)
from sklearn.calibration import calibration_curve
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
    threshold: float = 0.5,
) -> Dict:
    """
    Hitung semua metrics evaluasi.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Label aktual (0/1).
    y_pred : array-like, shape (n,)
        Label prediksi (0/1) pada threshold tertentu.
    y_prob : array-like, shape (n,)
        Probabilitas prediksi kelas 1 (inactive).
    model_name : str
        Nama model untuk logging.
    threshold : float
        Threshold yang digunakan untuk y_pred.

    Returns
    -------
    dict
        Dictionary berisi semua metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    n_total = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n_total - n_positive
    n_flagged = int(y_pred.sum())
    positive_rate_among_flagged = tp / n_flagged if n_flagged > 0 else 0
    recall_of_inactive = tp / n_positive if n_positive > 0 else 0

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence_pct": round(n_positive / n_total * 100, 2),
        # Core metrics
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        # Ranking metrics
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        # Calibration
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        # Confusion matrix
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # Business metrics
        "n_flagged": n_flagged,
        "pct_flagged": round(n_flagged / n_total * 100, 2),
        "positive_rate_among_flagged_pct": round(positive_rate_among_flagged * 100, 2),
        "recall_of_inactive_pct": round(recall_of_inactive * 100, 2),
    }

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
) -> float:
    """
    Temukan threshold optimal dari validation set.

    Parameters
    ----------
    strategy : str
        "f1" — maksimalkan F1 score
        "balanced" — dekat dengan 0.5 balanced accuracy
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if strategy == "f1":
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])
    else:
        return 0.5


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str,
):
    """Plot dan simpan confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Active (0)", "Inactive (1)"],
        yticklabels=["Active (0)", "Inactive (1)"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_ylabel("Label Aktual")
    ax.set_xlabel("Label Prediksi")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    results: Dict,
    y_true: np.ndarray,
    save_path: str,
):
    """
    Plot PR curves untuk semua model dalam satu plot.

    Parameters
    ----------
    results : dict
        Dict {model_name: y_prob_array}
    y_true : array
        Label aktual.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline_prevalence = y_true.mean()
    ax.axhline(baseline_prevalence, linestyle="--", color="gray", linewidth=1,
               label=f"Majority Class Baseline ({baseline_prevalence:.2f})")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (name, y_prob) in enumerate(results.items()):
        p, r, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        ax.plot(r, p, linewidth=2, color=colors[i], label=f"{name} (PR-AUC={auc_pr:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Semua Model", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    results: Dict,
    y_true: np.ndarray,
    save_path: str,
):
    """Plot ROC curves untuk semua model."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (name, y_prob) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, color=colors[i], label=f"{name} (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Semua Model", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_calibration(
    results: Dict,
    y_true: np.ndarray,
    save_path: str,
):
    """Plot calibration curves."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (name, y_prob) in enumerate(results.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)
        ax.plot(prob_pred, prob_true, "s-", linewidth=2, color=colors[i],
                label=f"{name} (Brier={brier:.3f})")

    ax.set_xlabel("Probabilitas Prediksi")
    ax.set_ylabel("Proporsi Aktual Positif")
    ax.set_title("Calibration Plot — Semua Model", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def print_comparison_table(all_metrics: list):
    """Print tabel perbandingan semua model."""
    df = pd.DataFrame(all_metrics)
    cols = ["model_name", "roc_auc", "pr_auc", "f1", "precision", "recall",
            "balanced_accuracy", "brier_score", "n_flagged", "pct_flagged"]
    df_display = df[[c for c in cols if c in df.columns]].copy()
    df_display = df_display.set_index("model_name")

    print("\n" + "="*90)
    print("PERBANDINGAN SEMUA MODEL (TEST SET)")
    print("="*90)
    print(df_display.to_string())
    print("="*90)
    return df_display
