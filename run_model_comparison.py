"""
5x5 Repeated Stratified Cross-Validation Benchmark on All 395 Students
Evaluates:
- Logistic / Ridge
- KNN
- SVM / SVR (RBF)
- Decision Tree
- Gradient Boosting
- Random Forest
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.metrics import (
    recall_score, roc_auc_score, average_precision_score,
    r2_score, mean_absolute_error, root_mean_squared_error
)

from data_loader import load_dataset
from feature_engineering import engineer_features

# 1. Load data
df = load_dataset("mat")
X, y_class, y_reg = engineer_features(df, include_g1=True)

print(f"Features: {X.shape[1]}, Total Samples: {len(X)}")

# 2. 5x5 Repeated Stratified Cross-Validation (25 iterations total)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

classifiers = {
    "Logistic / Ridge": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM / SVR (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=4, max_features=0.4, class_weight="balanced", random_state=42)
}

regressors = {
    "Logistic / Ridge": make_pipeline(StandardScaler(), Ridge(random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    "SVM / SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, max_features=0.7, criterion="squared_error", random_state=42)
}

results = {name: {"recall": [], "roc_auc": [], "avg_prec": [], "r2": [], "mae": [], "rmse": []} for name in classifiers}

for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_class)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    yc_train, yc_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    for name in classifiers:
        # Classification
        clf = classifiers[name]
        clf.fit(X_train, yc_train)
        pred_c = clf.predict(X_test)
        prob_c = clf.predict_proba(X_test)[:, 1]

        results[name]["recall"].append(recall_score(yc_test, pred_c, zero_division=0))
        results[name]["roc_auc"].append(roc_auc_score(yc_test, prob_c))
        results[name]["avg_prec"].append(average_precision_score(yc_test, prob_c))

        # Regression
        reg = regressors[name]
        reg.fit(X_train, yr_train)
        pred_r = reg.predict(X_test)

        results[name]["r2"].append(r2_score(yr_test, pred_r))
        results[name]["mae"].append(mean_absolute_error(yr_test, pred_r))
        results[name]["rmse"].append(root_mean_squared_error(yr_test, pred_r))

print("\n" + "="*80)
print("TABLE 4.5: MODEL COMPARISON (5x5 Repeated CV, Mean +/- SD)")
print("="*80)

summary_rows = []
for name in classifiers:
    rec_m, rec_s = np.mean(results[name]["recall"]), np.std(results[name]["recall"])
    auc_m, auc_s = np.mean(results[name]["roc_auc"]), np.std(results[name]["roc_auc"])
    ap_m, ap_s = np.mean(results[name]["avg_prec"]), np.std(results[name]["avg_prec"])
    r2_m, r2_s = np.mean(results[name]["r2"]), np.std(results[name]["r2"])
    mae_m, mae_s = np.mean(results[name]["mae"]), np.std(results[name]["mae"])
    rmse_m, rmse_s = np.mean(results[name]["rmse"]), np.std(results[name]["rmse"])

    summary_rows.append({
        "Model": name,
        "Recall": f"{rec_m:.3f} ± {rec_s:.3f}",
        "ROC-AUC": f"{auc_m:.3f} ± {auc_s:.3f}",
        "Avg. Precision": f"{ap_m:.3f} ± {ap_s:.3f}",
        "R2": f"{r2_m:.3f} ± {r2_s:.3f}",
        "MAE": f"{mae_m:.3f} ± {mae_s:.3f}",
        "raw_rec": rec_m,
        "raw_auc": auc_m,
        "raw_ap": ap_m,
        "raw_r2": r2_m,
        "raw_mae": mae_m,
        "raw_rmse": rmse_m
    })

res_df = pd.DataFrame(summary_rows)
print(res_df[["Model", "Recall", "ROC-AUC", "Avg. Precision", "R2", "MAE"]].to_string(index=False))

# Deliberate pairwise comparisons for the text
rf = res_df[res_df["Model"] == "Random Forest"].iloc[0]
dt = res_df[res_df["Model"] == "Decision Tree"].iloc[0]
gb = res_df[res_df["Model"] == "Gradient Boosting"].iloc[0]

auc_dt_diff = rf["raw_auc"] - dt["raw_auc"]
rmse_dt_diff = dt["raw_rmse"] - rf["raw_rmse"]
gb_auc_diff = rf["raw_auc"] - gb["raw_auc"]

print("\n--- DETAILED METRIC DELTAS FOR SECTION 4.5 TEXT ---")
print(f"RF Recall: {rf['Recall']}")
print(f"RF ROC-AUC: {rf['ROC-AUC']}")
print(f"RF MAE: {rf['MAE']}")
print(f"Decision Tree ROC-AUC improvement by RF: {auc_dt_diff:.3f}")
print(f"Decision Tree RMSE reduction by RF: {rmse_dt_diff:.3f} points")
print(f"RF vs Gradient Boosting ROC-AUC difference: {gb_auc_diff:.3f}")
