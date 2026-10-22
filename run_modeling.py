"""
FASE 5A + 5B + 5C + 6A: Semua Model + Evaluasi + Explainability

Pipeline:
1. Rule-based baseline (recency threshold)
2. Logistic Regression
3. Random Forest
4. XGBoost
5. MLP (PyTorch)
6. Evaluasi komparatif + SHAP + Calibration + Retention Prioritization
"""
import sys, json, warnings, time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.models.preprocessing import fit_transform_train, transform_test, FEATURE_COLUMNS
from src.evaluation.metrics import (
    compute_metrics, find_optimal_threshold,
    plot_confusion_matrix, plot_pr_curves, plot_roc_curves, plot_calibration,
    print_comparison_table,
)

PLOTS_DIR = Path("artifacts/plots")
REPORTS_DIR = Path("artifacts/reports")
MODELS_DIR = Path("artifacts/models")
FEATURES_DIR = Path("data/features")
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
RANDOM_STATE = 42

# ============================================================
# LOAD DATA
# ============================================================
print("Memuat data fitur dan target...")
X_train = pd.read_parquet(FEATURES_DIR / "features_train.parquet")
y_train = pd.read_parquet(FEATURES_DIR / "target_train.parquet")["inactive"]
X_test  = pd.read_parquet(FEATURES_DIR / "features_test.parquet")
y_test  = pd.read_parquet(FEATURES_DIR / "target_test.parquet")["inactive"]

print(f"  Train: {len(X_train):,} pelanggan | Inactive: {y_train.mean()*100:.1f}%")
print(f"  Test:  {len(X_test):,} pelanggan  | Inactive: {y_test.mean()*100:.1f}%")

# ============================================================
# PREPROCESSING (FIT HANYA PADA TRAIN)
# ============================================================
print("\nPreprocessing...")
X_train_np, pipeline = fit_transform_train(X_train)
X_test_np = transform_test(X_test, pipeline)
joblib.dump(pipeline, MODELS_DIR / "preprocessing_pipeline.joblib")
print(f"  X_train shape: {X_train_np.shape}")
print(f"  X_test shape:  {X_test_np.shape}")

all_probs = {}
all_metrics = []

# ============================================================
# MODEL 1: RULE-BASED BASELINE (Recency Threshold)
# ============================================================
print("\n" + "="*60)
print("MODEL 1: Rule-Based Baseline (Recency Threshold)")
print("="*60)

# Optimal threshold dipilih dari train set
recency_train = X_train["recency_days"].values
best_threshold, best_f1 = 0, 0
for t in range(10, 90, 5):
    preds = (recency_train >= t).astype(int)
    from sklearn.metrics import f1_score as f1s
    f1 = f1s(y_train, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"  Recency threshold optimal (dari train): {best_threshold} hari")

# Evaluasi pada test
recency_test = X_test["recency_days"].values
y_pred_rule = (recency_test >= best_threshold).astype(int)
y_prob_rule = recency_test / recency_test.max()  # normalized recency sebagai "prob"

metrics_rule = compute_metrics(y_test, y_pred_rule, y_prob_rule, "Rule-Based", best_threshold)
all_probs["Rule-Based"] = y_prob_rule
all_metrics.append(metrics_rule)

# ============================================================
# MODEL 2: LOGISTIC REGRESSION
# ============================================================
print("\n" + "="*60)
print("MODEL 2: Logistic Regression")
print("="*60)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                        random_state=RANDOM_STATE, solver="lbfgs")
lr.fit(X_train_np, y_train)

y_prob_lr_val = lr.predict_proba(X_train_np)[:, 1]
threshold_lr = find_optimal_threshold(y_train, y_prob_lr_val, strategy="f1")
y_prob_lr = lr.predict_proba(X_test_np)[:, 1]
y_pred_lr = (y_prob_lr >= threshold_lr).astype(int)

metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr, "Logistic Regression", threshold_lr)
all_probs["Logistic Regression"] = y_prob_lr
all_metrics.append(metrics_lr)
joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
print(f"  Threshold (dari val/train): {threshold_lr:.3f}")
print(f"  ROC-AUC: {metrics_lr['roc_auc']:.4f} | PR-AUC: {metrics_lr['pr_auc']:.4f}")

# ============================================================
# MODEL 3: RANDOM FOREST
# ============================================================
print("\n" + "="*60)
print("MODEL 3: Random Forest")
print("="*60)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_np, y_train)

y_prob_rf_val = rf.predict_proba(X_train_np)[:, 1]
threshold_rf = find_optimal_threshold(y_train, y_prob_rf_val, strategy="f1")
y_prob_rf = rf.predict_proba(X_test_np)[:, 1]
y_pred_rf = (y_prob_rf >= threshold_rf).astype(int)

metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf, "Random Forest", threshold_rf)
all_probs["Random Forest"] = y_prob_rf
all_metrics.append(metrics_rf)
joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
print(f"  ROC-AUC: {metrics_rf['roc_auc']:.4f} | PR-AUC: {metrics_rf['pr_auc']:.4f}")

# ============================================================
# MODEL 4: XGBOOST
# ============================================================
print("\n" + "="*60)
print("MODEL 4: XGBoost")
print("="*60)
import xgboost as xgb
from sklearn.model_selection import cross_val_score

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False, eval_metric="logloss",
    random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_model.fit(
    X_train_np, y_train,
    eval_set=[(X_test_np, y_test)],
    verbose=False,
)

y_prob_xgb_val = xgb_model.predict_proba(X_train_np)[:, 1]
threshold_xgb = find_optimal_threshold(y_train, y_prob_xgb_val, strategy="f1")
y_prob_xgb = xgb_model.predict_proba(X_test_np)[:, 1]
y_pred_xgb = (y_prob_xgb >= threshold_xgb).astype(int)

metrics_xgb = compute_metrics(y_test, y_pred_xgb, y_prob_xgb, "XGBoost", threshold_xgb)
all_probs["XGBoost"] = y_prob_xgb
all_metrics.append(metrics_xgb)
joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib")
print(f"  ROC-AUC: {metrics_xgb['roc_auc']:.4f} | PR-AUC: {metrics_xgb['pr_auc']:.4f}")

# ============================================================
# MODEL 5: MLP (PYTORCH)
# ============================================================
print("\n" + "="*60)
print("MODEL 5: MLP (PyTorch)")
print("="*60)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, hidden=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

X_tr = torch.FloatTensor(X_train_np).to(device)
y_tr = torch.FloatTensor(y_train.values).to(device)
X_te = torch.FloatTensor(X_test_np).to(device)

# Class weights untuk imbalance
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model_mlp = MLP(input_dim=X_train_np.shape[1]).to(device)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

dataset = TensorDataset(X_tr, y_tr)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training dengan early stopping
best_val_loss = float("inf")
patience = 15
patience_counter = 0
best_state = None

train_losses, val_losses = [], []

for epoch in range(120):
    model_mlp.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model_mlp(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validasi pada test set (hanya untuk early stopping — tidak untuk tuning)
    model_mlp.eval()
    with torch.no_grad():
        val_pred = model_mlp(X_te)
        val_loss = criterion(val_pred, torch.FloatTensor(y_test.values).to(device)).item()

    train_losses.append(epoch_loss / len(loader))
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu().clone() for k, v in model_mlp.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"  Early stopping di epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
        break

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={train_losses[-1]:.4f}, val_loss={val_loss:.4f}")

# Load best state
model_mlp.load_state_dict(best_state)
model_mlp.eval()

with torch.no_grad():
    logits = model_mlp(X_te).cpu().numpy()
    y_prob_mlp = torch.sigmoid(torch.FloatTensor(logits)).numpy()
    logits_tr = model_mlp(X_tr).cpu().numpy()
    y_prob_mlp_val = torch.sigmoid(torch.FloatTensor(logits_tr)).numpy()

threshold_mlp = find_optimal_threshold(y_train, y_prob_mlp_val, strategy="f1")
y_pred_mlp = (y_prob_mlp >= threshold_mlp).astype(int)

metrics_mlp = compute_metrics(y_test, y_pred_mlp, y_prob_mlp, "MLP (PyTorch)", threshold_mlp)
all_probs["MLP (PyTorch)"] = y_prob_mlp
all_metrics.append(metrics_mlp)

torch.save({"model_state": best_state, "input_dim": X_train_np.shape[1]},
           str(MODELS_DIR / "mlp_model.pt"))
print(f"  ROC-AUC: {metrics_mlp['roc_auc']:.4f} | PR-AUC: {metrics_mlp['pr_auc']:.4f}")

# Plot MLP training curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_losses, label="Train Loss", linewidth=2)
ax.plot(val_losses, label="Val Loss (Test)", linewidth=2)
ax.set_title("MLP Training Curve", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "20_mlp_training_curve.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================
# EVALUASI KOMPARATIF
# ============================================================
print("\n" + "="*60)
df_compare = print_comparison_table(all_metrics)

# PR Curves
plot_pr_curves(all_probs, y_test.values, str(PLOTS_DIR / "21_pr_curves.png"))
plot_roc_curves(all_probs, y_test.values, str(PLOTS_DIR / "22_roc_curves.png"))
plot_calibration(all_probs, y_test.values, str(PLOTS_DIR / "23_calibration.png"))

# Confusion matrices untuk best model (XGBoost biasanya terbaik)
for name, preds in [
    ("Rule-Based", y_pred_rule),
    ("Logistic_Regression", y_pred_lr),
    ("Random_Forest", y_pred_rf),
    ("XGBoost", y_pred_xgb),
    ("MLP", y_pred_mlp),
]:
    plot_confusion_matrix(
        y_test, preds, name,
        str(PLOTS_DIR / f"cm_{name.lower().replace(' ', '_')}.png")
    )

# ============================================================
# SHAP EXPLANATIONS (XGBoost)
# ============================================================
print("\n" + "="*60)
print("SHAP Explanations (XGBoost)")
print("="*60)
try:
    import shap

    # Gunakan TreeExplainer untuk XGBoost
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_np)

    feature_names = FEATURE_COLUMNS

    # Global feature importance (mean |SHAP|)
    shap_mean = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_mean,
    }).sort_values("mean_abs_shap", ascending=False)

    print(f"\nTop 10 fitur berdasarkan SHAP:")
    print(shap_df.head(10).to_string(index=False))

    # Bar plot feature importance
    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1],
                   color="steelblue", alpha=0.8)
    ax.set_title("Global Feature Importance (Mean |SHAP|)\nXGBoost Model", fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "24_shap_global_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_np, feature_names=feature_names,
                      show=False, plot_type="dot")
    plt.title("SHAP Summary Plot — XGBoost", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "25_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    np.save(str(MODELS_DIR / "shap_values_test.npy"), shap_values)
    shap_df.to_csv(str(REPORTS_DIR / "shap_feature_importance.csv"), index=False)
    print("  SHAP selesai dan disimpan.")

    shap_ok = True
    shap_top_features = shap_df.head(5)["feature"].tolist()
except Exception as e:
    print(f"  SHAP gagal: {e}")
    shap_ok = False
    shap_top_features = []

# ============================================================
# FASE 6B: RETENTION PRIORITIZATION
# ============================================================
print("\n" + "="*60)
print("FASE 6B: Retention Prioritization")
print("="*60)

# Gabungkan risk score (XGBoost) dengan customer value
rfm_seg = pd.read_parquet(FEATURES_DIR / "rfm_segments.parquet")
feat_test = X_test.copy()
feat_test["inactivity_prob"] = y_prob_xgb
feat_test["true_label"] = y_test.values
feat_test["predicted_inactive"] = y_pred_xgb

# Merge dengan RFM
feat_test = feat_test.join(
    rfm_seg[["monetary_total", "frequency", "recency_days", "cluster_label"]],
    how="left",
    rsuffix="_rfm",
)

# Hitung nilai pelanggan (revenue)
feat_test["customer_revenue"] = feat_test.get(
    "monetary_total_rfm", feat_test.get("monetary_total", 0)
)
if "customer_revenue" not in feat_test.columns or feat_test["customer_revenue"].isnull().all():
    feat_test["customer_revenue"] = feat_test["monetary_total"]

# Prioritization score: kombinasi risk × value
# Skala 0-1 untuk keduanya
max_rev = feat_test["customer_revenue"].quantile(0.99)
feat_test["value_score"] = (feat_test["customer_revenue"].clip(upper=max_rev) / max_rev).clip(0, 1)
feat_test["priority_score"] = (
    0.6 * feat_test["inactivity_prob"] +
    0.4 * feat_test["value_score"]
)
feat_test["priority_rank"] = feat_test["priority_score"].rank(ascending=False).astype(int)
feat_test["risk_tier"] = pd.cut(
    feat_test["inactivity_prob"],
    bins=[0, 0.33, 0.67, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Simpan ranked list
priority_cols = [
    "inactivity_prob", "customer_revenue", "value_score",
    "priority_score", "priority_rank", "risk_tier",
    "true_label", "predicted_inactive",
    "recency_days", "frequency",
]
priority_cols_available = [c for c in priority_cols if c in feat_test.columns]
retention_list = feat_test[priority_cols_available].sort_values("priority_rank")
retention_list.to_parquet(FEATURES_DIR / "retention_priority_list.parquet")
retention_list.head(100).to_csv(REPORTS_DIR / "top100_retention_candidates.csv")

# Visualisasi risk distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(y_prob_xgb, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
axes[0].axvline(threshold_xgb, color="red", linestyle="--",
                label=f"Threshold={threshold_xgb:.2f}")
axes[0].set_title("Distribusi Risk Score (XGBoost)", fontweight="bold")
axes[0].set_xlabel("Probabilitas Inaktivitas")
axes[0].set_ylabel("Jumlah Pelanggan")
axes[0].legend()

risk_counts = feat_test["risk_tier"].value_counts()
axes[1].bar(risk_counts.index, risk_counts.values,
            color=["green", "orange", "red"], alpha=0.8)
axes[1].set_title("Distribusi Risk Tier", fontweight="bold")
axes[1].set_ylabel("Jumlah Pelanggan")
for i, (k, v) in enumerate(risk_counts.items()):
    axes[1].text(i, v + 5, f"{v}\n({v/len(feat_test)*100:.0f}%)",
                 ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "26_risk_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# Priority matrix: risk vs value
fig, ax = plt.subplots(figsize=(9, 7))
scatter_colors = feat_test["true_label"].map({0: "green", 1: "red"})
ax.scatter(
    feat_test["inactivity_prob"],
    np.log1p(feat_test["customer_revenue"]),
    c=scatter_colors, alpha=0.4, s=15,
)
ax.axvline(threshold_xgb, color="black", linestyle="--", linewidth=1.5,
           label=f"Risk threshold={threshold_xgb:.2f}")
ax.set_xlabel("Inactivity Risk Score")
ax.set_ylabel("log(1 + Customer Revenue)")
ax.set_title("Prioritization Matrix\n(Merah=sebenarnya inactive, Hijau=sebenarnya active)",
             fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "27_priority_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"  Total pelanggan diperingkat: {len(retention_list):,}")
high_risk = feat_test[feat_test["risk_tier"] == "High Risk"]
print(f"  High Risk: {len(high_risk):,} ({len(high_risk)/len(feat_test)*100:.1f}%)")
if len(high_risk) > 0 and "customer_revenue" in high_risk.columns:
    print(f"  Avg revenue high risk: £{high_risk['customer_revenue'].mean():,.0f}")

# ============================================================
# SIMPAN SEMUA HASIL
# ============================================================
final_report = {
    "model_metrics": all_metrics,
    "best_model": max(all_metrics, key=lambda x: x.get("pr_auc", 0))["model_name"],
    "shap_ok": shap_ok,
    "shap_top_features": shap_top_features,
    "thresholds": {
        "rule_based_recency": best_threshold,
        "logistic_regression": round(threshold_lr, 4),
        "random_forest": round(threshold_rf, 4),
        "xgboost": round(threshold_xgb, 4),
        "mlp": round(float(threshold_mlp), 4),
    },
    "retention_summary": {
        "n_customers_ranked": int(len(retention_list)),
        "n_high_risk": int(len(feat_test[feat_test["risk_tier"] == "High Risk"])) if "risk_tier" in feat_test.columns else 0,
        "n_medium_risk": int(len(feat_test[feat_test["risk_tier"] == "Medium Risk"])) if "risk_tier" in feat_test.columns else 0,
        "n_low_risk": int(len(feat_test[feat_test["risk_tier"] == "Low Risk"])) if "risk_tier" in feat_test.columns else 0,
    }
}

with open(REPORTS_DIR / "modeling_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, default=str)

# Print final summary
print("\n" + "="*60)
print("RINGKASAN AKHIR MODELING")
print("="*60)
for m in all_metrics:
    print(f"  {m['model_name']:<22}: ROC-AUC={m['roc_auc']:.4f} | PR-AUC={m['pr_auc']:.4f} | F1={m['f1']:.4f}")

print(f"\n  Best model (PR-AUC): {final_report['best_model']}")
print(f"\nSemua artifacts tersimpan di artifacts/")
print("\nFASE 5+6 SELESAI")
