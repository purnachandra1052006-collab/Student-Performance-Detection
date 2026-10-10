"""
Feature Engineering and Preprocessing for Early Student Performance Prediction
Includes domain-specific behavioral composites, non-linear grade transforms, and momentum indicators.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def engineer_features(df: pd.DataFrame, include_g1: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs feature engineering and splits features from targets.

    Parameters:
        df (pd.DataFrame): Raw student dataset.
        include_g1 (bool): If True, include Period 1 grade (G1) and derived early momentum signals.

    Returns:
        X (pd.DataFrame): Processed feature matrix.
        y_class (pd.Series): Binary target (1 = At-Risk [G3 < 10], 0 = Passing [G3 >= 10]).
        y_reg (pd.Series): Continuous target (Final Grade G3, 0-20).
    """
    data = df.copy()

    # 1. Targets
    y_reg = data["G3"].astype(float)
    y_class = (data["G3"] < 10).astype(int)  # 1 = At-Risk, 0 = Passing

    # 2. Domain-Specific Behavioral & Demographic Composites
    data["total_alcohol"] = data["Dalc"] + data["Walc"]
    data["social_risk"] = data["total_alcohol"] * data["goout"]
    data["parent_edu_total"] = data["Medu"] + data["Fedu"]
    data["parent_edu_interaction"] = data["Medu"] * data["Fedu"]
    data["study_free_ratio"] = data["studytime"] / (data["freetime"] + 0.1)
    data["has_prior_failures"] = (data["failures"] > 0).astype(int)
    data["high_absences"] = (data["absences"] > 5).astype(int)
    data["log_absences"] = np.log1p(data["absences"])

    # Support Network Aggregate
    support_score = 0
    if "schoolsup" in data.columns:
        support_score += (data["schoolsup"] == "yes").astype(int)
    if "famsup" in data.columns:
        support_score += (data["famsup"] == "yes").astype(int)
    if "paid" in data.columns:
        support_score += (data["paid"] == "yes").astype(int)
    data["support_network_score"] = support_score

    # 3. Non-Linear Early Academic Velocity & Threshold Signals (G1)
    if include_g1 and "G1" in data.columns:
        data["G1_failing"] = (data["G1"] < 10).astype(int)
        data["G1_honors"] = (data["G1"] >= 14).astype(int)
        data["G1_squared"] = data["G1"] ** 2
        data["academic_momentum"] = data["G1"] - (data["failures"] * 3.0)
        data["risk_index"] = (
            (data["failures"] * 2.0)
            + (data["G1"] < 10).astype(int) * 3.0
            + (data["absences"] > 5).astype(int) * 1.5
            - (data["studytime"] >= 3).astype(int) * 1.0
        )

    # 4. Drop outcome/late-stage variables to ensure strict Early Prediction
    drop_cols = ["G3", "G2"]
    if not include_g1 and "G1" in data.columns:
        drop_cols.append("G1")

    features = data.drop(columns=[col for col in drop_cols if col in data.columns])

    # 5. One-Hot Encoding for categorical features
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

    return X, y_class, y_reg


def prepare_train_test_data(
    df: pd.DataFrame,
    include_g1: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits features and targets into reproducible stratified train and test sets.
    """
    X, y_class, y_reg = engineer_features(df, include_g1=include_g1)

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_class_train, y_class_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
    y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]

    return (
        X_train, X_test,
        y_class_train, y_class_test,
        y_reg_train, y_reg_test,
        X.columns.tolist()
    )
