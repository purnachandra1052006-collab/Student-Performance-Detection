"""
High-Performance Model Suite: Early Student Performance Classification & Regression
Architecture: Tuned Random Forest Ensembles (Classifier + Regressor)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_absolute_error, root_mean_squared_error, r2_score
)


class EarlyStudentPerformanceModel:
    """
    High-Performance Random Forest model suite for Early Student Performance Prediction:
    - Classification: Tuned At-Risk student identification (max_depth=4, max_features=0.4, class_weight='balanced')
    - Regression: Tuned Final Grade G3 prediction (max_depth=10, max_features=0.7, criterion='squared_error')
    - Explainability & Interventions: Multi-factor risk attribution & prescriptive action generation.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self.clf = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators_clf", 150),
            max_depth=kwargs.get("max_depth_clf", 4),
            max_features=kwargs.get("max_features_clf", 0.4),
            class_weight="balanced",
            min_samples_leaf=1,
            random_state=random_state
        )
        self.reg = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators_reg", 150),
            max_depth=kwargs.get("max_depth_reg", 10),
            max_features=kwargs.get("max_features_reg", 0.7),
            criterion="squared_error",
            min_samples_leaf=1,
            random_state=random_state
        )
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_class_train: pd.Series, y_reg_train: pd.Series):
        """Fits both classification and regression models."""
        self.feature_names = list(X_train.columns)
        self.clf.fit(X_train, y_class_train)
        self.reg.fit(X_train, y_reg_train)
        return self

    def evaluate_classification(self, X_test: pd.DataFrame, y_class_test: pd.Series) -> dict:
        """Evaluates at-risk classification performance."""
        y_pred = self.clf.predict(X_test)
        y_prob = self.clf.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_class_test, y_pred),
            "Precision": precision_score(y_class_test, y_pred, zero_division=0),
            "Recall": recall_score(y_class_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_class_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_class_test, y_prob),
            "Confusion Matrix": confusion_matrix(y_class_test, y_pred).tolist(),
            "Report": classification_report(y_class_test, y_pred, target_names=["Passing", "At-Risk"], output_dict=True)
        }
        return metrics

    def evaluate_regression(self, X_test: pd.DataFrame, y_reg_test: pd.Series) -> dict:
        """Evaluates final grade regression performance."""
        y_pred = self.reg.predict(X_test)

        metrics = {
            "MAE (Mean Absolute Error)": mean_absolute_error(y_reg_test, y_pred),
            "RMSE (Root Mean Squared Error)": root_mean_squared_error(y_reg_test, y_pred),
            "R2 Score": r2_score(y_reg_test, y_pred)
        }
        return metrics

    def get_feature_importances(self, top_n: int = 10) -> pd.DataFrame:
        """Returns top influential features for classification and regression."""
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Classification_Importance": self.clf.feature_importances_,
            "Regression_Importance": self.reg.feature_importances_
        })
        df_imp["Combined_Importance"] = (df_imp["Classification_Importance"] + df_imp["Regression_Importance"]) / 2
        return df_imp.sort_values(by="Combined_Importance", ascending=False).head(top_n).reset_index(drop=True)

    def predict_student(self, student_features: pd.DataFrame) -> dict:
        """
        Predicts at-risk status, grade, and generates actionable intervention guidance.
        """
        at_risk_prob = float(self.clf.predict_proba(student_features)[:, 1][0])
        is_at_risk = bool(self.clf.predict(student_features)[0] == 1)
        predicted_grade = float(self.reg.predict(student_features)[0])

        interventions = []
        student_data = student_features.iloc[0]

        if "failures" in student_data and student_data["failures"] > 0:
            interventions.append("Tutoring: Prior academic failure detected; assign weekly 1-on-1 remedial tutoring.")
        
        if "absences" in student_data and student_data["absences"] > 5:
            interventions.append(f"Attendance: Chronic absenteeism ({int(student_data['absences'])} absences); initiate parent outreach and attendance contract.")
        
        if "study_free_ratio" in student_data and student_data["study_free_ratio"] < 0.35:
            interventions.append("Study Habits: Low study-to-leisure ratio; provide academic time-management coaching.")
        
        if "total_alcohol" in student_data and student_data["total_alcohol"] >= 4:
            interventions.append("Wellness: Elevated alcohol consumption flagged; refer to student counseling and wellbeing mentor.")
            
        if "schoolsup_yes" in student_data and student_data["schoolsup_yes"] == 0 and is_at_risk:
            interventions.append("Academic Support: Enroll student in institutional extra educational support program.")

        if not interventions and is_at_risk:
            interventions.append("General Intervention: Schedule academic advisor consultation for tailored study plan.")
        elif not is_at_risk and not interventions:
            interventions.append("Positive Reinforcement: Student on track. Encourage advanced coursework or peer tutoring role.")

        return {
            "At-Risk Status": "AT-RISK" if is_at_risk else "SAFE / PASSING",
            "At-Risk Probability": f"{at_risk_prob * 100:.1f}%",
            "Predicted Final Grade (0-20)": round(predicted_grade, 2),
            "Recommended Interventions": interventions
        }
