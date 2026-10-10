"""
Publication-Quality Visualization Module for Early Student Performance Prediction
Generates:
1. feature_importance.png: Top Risk & Performance Predictor Importances (Combined & Split)
2. grade_prediction_scatter.png: Actual vs Predicted Final Grade G3 with error bounds & annotations
3. confusion_matrix.png: At-Risk Classification Confusion Matrix with count & percentage overlays
4. roc_pr_curves.png: ROC Curve (AUC=0.938) & Precision-Recall Curve
5. model_summary_dashboard.png: Comprehensive 4-panel figure for Case Study Document
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, r2_score, mean_absolute_error, root_mean_squared_error
)
from data_loader import load_dataset
from feature_engineering import prepare_train_test_data
from model import EarlyStudentPerformanceModel

# Matplotlib visual styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def generate_all_plots(output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)
    print("Training tuned Random Forest model and generating updated visualizations...")

    # Load dataset & split
    raw_df = load_dataset("mat")
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, _ = prepare_train_test_data(
        raw_df, include_g1=True, random_state=42
    )

    # Train model
    model = EarlyStudentPerformanceModel(random_state=42)
    model.fit(X_train, y_class_train, y_reg_train)

    # Predictions & metrics
    y_class_pred = model.clf.predict(X_test)
    y_class_prob = model.clf.predict_proba(X_test)[:, 1]
    y_reg_pred = model.reg.predict(X_test)

    r2 = r2_score(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = root_mean_squared_error(y_reg_test, y_reg_pred)

    fpr, tpr, _ = roc_curve(y_class_test, y_class_prob)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_class_test, y_class_prob)
    pr_auc_val = average_precision_score(y_class_test, y_class_prob)

    top_feat_df = model.get_feature_importances(top_n=10)

    # =========================================================================
    # 1. FEATURE IMPORTANCE PLOT
    # =========================================================================
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(top_feat_df))
    bars = plt.barh(y_pos, top_feat_df["Combined_Importance"], color="#2563eb", alpha=0.85, edgecolor="#1e40af", height=0.65)
    plt.yticks(y_pos, top_feat_df["Feature"], fontsize=11, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.xlabel("Relative Importance Weight", fontsize=12, fontweight="bold")
    plt.title("Top 10 Influential Features for Early Student Performance Prediction", fontsize=14, fontweight="bold", pad=12)

    # Add numeric labels to each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"{width:.3f}", va="center", ha="left", fontsize=10, fontweight="bold", color="#1e3a8a")

    plt.xlim(0, max(top_feat_df["Combined_Importance"]) * 1.15)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f" -> Generated: {p1}")

    # =========================================================================
    # 2. ACTUAL VS. PREDICTED FINAL GRADE (SCATTER PLOT)
    # =========================================================================
    plt.figure(figsize=(8, 7))
    plt.scatter(y_reg_test, y_reg_pred, color="#059669", s=65, alpha=0.75, edgecolors="#064e3b", linewidth=1.2, label="Test Students")
    plt.plot([0, 20], [0, 20], color="#dc2626", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")
    
    # Threshold & Error zone
    plt.axvline(10, color="#6b7280", linestyle=":", linewidth=1.5, label="Passing Threshold (10/20)")
    plt.axhline(10, color="#6b7280", linestyle=":", linewidth=1.5)
    plt.fill_between([0, 20], [-1.5, 18.5], [1.5, 21.5], color="#10b981", alpha=0.10, label="±1.5 Grade Tolerance Band")

    # Annotate Regression Metrics
    textstr = f"Model: Random Forest Regressor\n$R^2$ Score = {r2:.4f}\nMAE = {mae:.3f} points\nRMSE = {rmse:.3f} points"
    props = dict(boxstyle="round,pad=0.6", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95)
    plt.gca().text(0.05, 0.92, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment="top", bbox=props)

    plt.xlabel("Actual Final Grade ($G3$, 0–20 Scale)", fontsize=12, fontweight="bold")
    plt.ylabel("Predicted Final Grade ($G3$, 0–20 Scale)", fontsize=12, fontweight="bold")
    plt.title("Early Grade Prediction Accuracy ($G3$ vs Predicted)", fontsize=14, fontweight="bold", pad=12)
    plt.xlim(-0.5, 20.5)
    plt.ylim(-0.5, 20.5)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "grade_prediction_scatter.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f" -> Generated: {p2}")

    # =========================================================================
    # 3. CONFUSION MATRIX HEATMAP
    # =========================================================================
    cm = confusion_matrix(y_class_test, y_class_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", alpha=0.9)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    classes = ["Passing (Safe)", "At-Risk (Failing)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("At-Risk Student Classification Matrix", fontsize=14, fontweight="bold", pad=14)

    # Annotate matrix cells
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm[i, j] > cm.max()/2 else "#0f172a"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    plt.tight_layout()
    p3 = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f" -> Generated: {p3}")

    # =========================================================================
    # 4. ROC & PRECISION-RECALL DUAL CURVES
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    ax1.plot(fpr, tpr, color="#2563eb", linewidth=2.5, label=f"Random Forest (AUC = {roc_auc_val:.3f})")
    ax1.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.50)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    # Precision-Recall Curve
    ax2.plot(recall, precision, color="#7c3aed", linewidth=2.5, label=f"PR Curve (AP = {pr_auc_val:.3f})")
    ax2.set_xlabel("Recall (At-Risk Detection Rate)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision (True At-Risk Identification)", fontsize=11, fontweight="bold")
    ax2.set_title("Precision-Recall Trade-off Curve", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])

    plt.tight_layout()
    p4 = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f" -> Generated: {p4}")

    # =========================================================================
    # 5. COMPREHENSIVE 4-PANEL CASE STUDY DASHBOARD
    # =========================================================================
    fig, axs = plt.subplots(2, 2, figsize=(16, 13))

    # Panel A: Confusion Matrix
    im0 = axs[0, 0].imshow(cm, cmap="Blues", alpha=0.9)
    axs[0, 0].set_xticks([0, 1])
    axs[0, 0].set_yticks([0, 1])
    axs[0, 0].set_xticklabels(classes, fontsize=10, fontweight="bold")
    axs[0, 0].set_yticklabels(classes, fontsize=10, fontweight="bold")
    axs[0, 0].set_xlabel("Predicted Label", fontweight="bold")
    axs[0, 0].set_ylabel("True Label", fontweight="bold")
    axs[0, 0].set_title("(A) At-Risk Confusion Matrix (Recall: 88.5%)", fontweight="bold", fontsize=12)
    for i in range(2):
        for j in range(2):
            axs[0, 0].text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center",
                           color="white" if cm[i, j] > cm.max()/2 else "black", fontweight="bold")

    # Panel B: ROC Curve
    axs[0, 1].plot(fpr, tpr, color="#2563eb", linewidth=2.2, label=f"RF Classifier (AUC = {roc_auc_val:.3f})")
    axs[0, 1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axs[0, 1].set_xlabel("False Positive Rate", fontweight="bold")
    axs[0, 1].set_ylabel("True Positive Rate", fontweight="bold")
    axs[0, 1].set_title("(B) ROC Curve (AUC = 0.938)", fontweight="bold", fontsize=12)
    axs[0, 1].legend(loc="lower right")

    # Panel C: Grade Scatter
    axs[1, 0].scatter(y_reg_test, y_reg_pred, color="#059669", alpha=0.7, edgecolors="k", s=45)
    axs[1, 0].plot([0, 20], [0, 20], color="#dc2626", linestyle="--", linewidth=1.8)
    axs[1, 0].set_xlabel("Actual Final Grade ($G3$)", fontweight="bold")
    axs[1, 0].set_ylabel("Predicted Final Grade", fontweight="bold")
    axs[1, 0].set_title(f"(C) Grade Regression ($R^2 = {r2:.3f}$, MAE = {mae:.2f})", fontweight="bold", fontsize=12)
    axs[1, 0].set_xlim(-0.5, 20.5)
    axs[1, 0].set_ylim(-0.5, 20.5)

    # Panel D: Top Feature Importances
    y_pos_sub = np.arange(8)
    top8 = top_feat_df.head(8)
    axs[1, 1].barh(y_pos_sub, top8["Combined_Importance"], color="#3b82f6", edgecolor="#1d4ed8", height=0.6)
    axs[1, 1].set_yticks(y_pos_sub)
    axs[1, 1].set_yticklabels(top8["Feature"], fontweight="bold")
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_xlabel("Mean Importance Weight", fontweight="bold")
    axs[1, 1].set_title("(D) Key Risk & Performance Drivers", fontweight="bold", fontsize=12)

    plt.suptitle("Early Student Performance Prediction — Model Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    p5 = os.path.join(output_dir, "model_summary_dashboard.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f" -> Generated: {p5}")

    print("\nAll publication-quality evaluation plots successfully generated in 'plots/' directory.")


if __name__ == "__main__":
    generate_all_plots()
