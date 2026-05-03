"""
Task 2: Classification of Transportation Modes
Assignment 2 – GPS Mobility Analysis using GeoLife Dataset
===========================================================
Models:  1. Random Forest Classifier
         2. Gradient Boosting Classifier
         3. MLP Neural Network (Multi-layer Perceptron)

Data split strategy:
  Stratified 80/20 split by mode class (walk/bus/bike/car).
  Stratification ensures every class is proportionally represented
  in both train and test sets – this is critical for achieving
  consistent accuracy across modes (>80% target).

Features: 8 literature-grounded features based on
  Li et al. (2021) Tsinghua Science & Technology, 26(4): 403-416.

Produces (in plots/):
  confusion_matrix_rf.png
  confusion_matrix_gb.png
  confusion_matrix_xgb.png
  metrics_comparison.png
  per_class_f1.png
  roc_curves.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, sys, time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
CSV   = os.path.join(BASE, "segment_features.csv")
os.makedirs(PLOTS, exist_ok=True)

# ── Palette / constants ───────────────────────────────────────────────────────
PALETTE   = {"walk": "#4CAF50", "bus": "#2196F3", "bike": "#FF9800", "car": "#F44336"}
CLASSES   = ["walk", "bus", "bike", "car"]
DROP_COLS = ["segment_id", "user_id", "file_name", "mode"]

# Aligning exactly with the 7 features finalised in Task 1
SELECTED_FEATURES = [
    "avg_speed_mps",   # Average speed
    "max_speed_mps",   # Maximum speed
    "distance_m",      # Distance travelled
    "std_acc_mps2",    # Acceleration variation
    "avg_turn_deg",    # Average turning angle
    "stop_ratio",      # Stop ratio
    "std_speed_mps",   # Speed variation
]

# ── Load & prepare data ───────────────────────────────────────────────────────
print("=" * 60)
print("Loading data …")
df       = pd.read_csv(CSV)
FEATURES = SELECTED_FEATURES   # 8 literature-grounded features

X = df[FEATURES].fillna(0).values

le = LabelEncoder()
le.fit(CLASSES)
y = le.transform(df["mode"])

print(f"  Total samples  : {len(X)}")
print(f"  Features used  : {len(FEATURES)} (literature-based)  →  {FEATURES}")
print(f"  Classes        : {le.classes_.tolist()}")
print(f"  Distribution   : {dict(zip(*np.unique(le.inverse_transform(y), return_counts=True)))}")

# ── 80/20 stratified split ────────────────────────────────────────────────────
# stratify=y  →  each class (walk/bus/bike/car) maintains its proportion
# in BOTH train and test sets.  This is essential for reliable accuracy;
# a purely random split risks under-representing minority classes in the
# test set, leading to inflated or misleading overall accuracy.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n  Train: {len(X_train)}  |  Test: {len(X_test)}")
print("  Train class counts:", dict(zip(CLASSES, [int((y_train==i).sum()) for i in range(len(CLASSES))])))
print("  Test  class counts:", dict(zip(CLASSES, [int((y_test ==i).sum()) for i in range(len(CLASSES))])))

# Scale features – MLP and distance-based models require normalisation;
# tree ensembles (RF, GB) are scale-invariant but we use the same data.
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, model_name, filename):
    """Annotated confusion matrix heatmap."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    path = os.path.join(PLOTS, filename)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def evaluate(name, y_true, y_pred):
    """Print metrics and return (accuracy, weighted-F1, classification report dict)."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    rep = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    print(f"\n{'─'*50}")
    print(f"  Model   : {name}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (wtd): {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES)}")
    return acc, f1, rep


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1 – Random Forest
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 1: Random Forest Classifier")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                             min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf, f1_rf, rep_rf = evaluate("Random Forest", y_test, y_pred_rf)
print(f"  Training time: {time.time()-t0:.1f}s")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf),
                      "Random Forest", "confusion_matrix_rf.png")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2 – Gradient Boosting
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2: Gradient Boosting Classifier")
t0 = time.time()
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=5, subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb, f1_gb, rep_gb = evaluate("Gradient Boosting", y_test, y_pred_gb)
print(f"  Training time: {time.time()-t0:.1f}s")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_gb),
                      "Gradient Boosting", "confusion_matrix_gb.png")


# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3 – XGBoost Classifier
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3: XGBoost Classifier")
t0 = time.time()
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False
)
# XGBoost can use the unscaled features (X_train), but we use X_train_s 
# here just for consistency with the evaluation function
xgb.fit(X_train_s, y_train)
y_pred_xgb = xgb.predict(X_test_s)
acc_xgb, f1_xgb, rep_xgb = evaluate("XGBoost", y_test, y_pred_xgb)
print(f"  Training time: {time.time()-t0:.1f}s")
plot_confusion_matrix(confusion_matrix(y_test, y_pred_xgb),
                      "XGBoost", "confusion_matrix_xgb.png")


# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════

models     = ["Random Forest", "Gradient Boosting", "XGBoost"]
accs       = [acc_rf, acc_gb, acc_xgb]
f1s        = [f1_rf, f1_gb, f1_xgb]
bar_colors = ["#2196F3", "#4CAF50", "#FF9800"]

# ── Overall Accuracy + F1 comparison ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, vals, metric_name in zip(axes, [accs, f1s], ["Accuracy", "Weighted F1-Score"]):
    bars = ax.bar(models, vals, color=bar_colors, edgecolor="white",
                  linewidth=1.4, width=0.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_title(f"Model Comparison – {metric_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel(metric_name, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.2%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

plt.suptitle("Classification Model Performance Summary (7 Features)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "metrics_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved: {os.path.join(PLOTS, 'metrics_comparison.png')}")

# ── Per-class F1 comparison ───────────────────────────────────────────────────
per_class_f1 = {
    "Random Forest":     [rep_rf[c]["f1-score"]  for c in CLASSES],
    "Gradient Boosting": [rep_gb[c]["f1-score"]  for c in CLASSES],
    "XGBoost":           [rep_xgb[c]["f1-score"] for c in CLASSES],
}
x     = np.arange(len(CLASSES))
width = 0.26

fig, ax = plt.subplots(figsize=(10, 6))
for i, (mname, vals) in enumerate(per_class_f1.items()):
    bars = ax.bar(x + (i - 1) * width, vals, width=width, label=mname,
                  color=bar_colors[i], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title("Per-Class F1-Score Comparison — 7 Features",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "per_class_f1.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(PLOTS, 'per_class_f1.png')}")

# ── One-vs-Rest ROC curves (Random Forest) ────────────────────────────────────
y_test_bin = label_binarize(y_test, classes=range(len(CLASSES)))
y_score_rf = rf.predict_proba(X_test)   # RF has predict_proba natively

fig, ax = plt.subplots(figsize=(8, 7))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
    ax.plot(fpr, tpr, color=PALETTE[cls], linewidth=2.2,
            label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random chance")
ax.set_xlim([-0.01, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – Random Forest (One-vs-Rest, 7 Features)",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "roc_curves.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.join(PLOTS, 'roc_curves.png')}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON SUMMARY  (7 Features)")
print(f"{'Model':<22} {'Accuracy':>10} {'F1 (wtd)':>12}")
print("─" * 46)
for m, a, f in zip(models, accs, f1s):
    print(f"  {m:<20} {a:>10.4f} {f:>12.4f}")
print("=" * 60)
print("\nTask 2 complete. All plots saved to plots/")
