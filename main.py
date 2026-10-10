"""
Main Execution Script for Early Student Performance Prediction
Runs end-to-end: Data Loading -> Feature Engineering -> ML Training -> Evaluation -> Explainability & Intervention
"""

import pandas as pd
from data_loader import load_dataset
from feature_engineering import prepare_train_test_data
from model import EarlyStudentPerformanceModel


def run_pipeline(subject: str = "mat", include_g1: bool = True):
    print("=" * 75)
    print(f" EARLY STUDENT PERFORMANCE PREDICTION (Subject: {subject.upper()})")
    print(f" Early Prediction Horizon: {'Using Early Term Exam (G1) + Profile' if include_g1 else 'Intake/Baseline Profile Only'}")
    print(" Model: Random Forest (Classification + Regression Ensemble)")
    print("=" * 75)

    # 1. Load Data
    raw_df = load_dataset(subject=subject)
    print(f"\n[1] Dataset loaded: {raw_df.shape[0]} students, {raw_df.shape[1]} raw attributes.")

    # 2. Feature Engineering & Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, feature_names = prepare_train_test_data(
        raw_df, include_g1=include_g1, test_size=0.2, random_state=42
    )
    print(f"[2] Feature Engineering complete: {X_train.shape[1]} engineered features.")
    print(f"    - Training set: {X_train.shape[0]} samples")
    print(f"    - Testing set : {X_test.shape[0]} samples")
    print(f"    - Test at-risk base rate: {y_class_test.mean():.1%}")

    # 3. Model Training
    print("\n[3] Training Random Forest Models...")
    model = EarlyStudentPerformanceModel(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_class_train, y_reg_train)
    print("    Training completed successfully.")

    # 4. Classification Evaluation
    print("\n" + "-" * 40)
    print(" TASK 1: CLASSIFICATION (At-Risk vs. Safe)")
    print("-" * 40)
    class_metrics = model.evaluate_classification(X_test, y_class_test)
    print(f"  * Accuracy : {class_metrics['Accuracy']:.4f}")
    print(f"  * Precision: {class_metrics['Precision']:.4f}")
    print(f"  * Recall   : {class_metrics['Recall']:.4f}")
    print(f"  * F1-Score : {class_metrics['F1-Score']:.4f}")
    print(f"  * ROC-AUC  : {class_metrics['ROC-AUC']:.4f}")
    print("  * Confusion Matrix [TN, FP], [FN, TP]:", class_metrics["Confusion Matrix"])

    # 5. Regression Evaluation
    print("\n" + "-" * 40)
    print(" TASK 2: REGRESSION (Predicted Final Grade 0-20)")
    print("-" * 40)
    reg_metrics = model.evaluate_regression(X_test, y_reg_test)
    print(f"  * MAE  (Mean Absolute Error)     : {reg_metrics['MAE (Mean Absolute Error)']:.3f} points")
    print(f"  * RMSE (Root Mean Squared Error) : {reg_metrics['RMSE (Root Mean Squared Error)']:.3f} points")
    print(f"  * R2 Score                       : {reg_metrics['R2 Score']:.4f}")

    # 6. Explainability: Top Feature Importances
    print("\n" + "-" * 40)
    print(" TASK 3: EXPLAINABILITY & KEY RISK FACTORS")
    print("-" * 40)
    top_features = model.get_feature_importances(top_n=8)
    print(top_features.to_string(index=False))

    # 7. Student Case Studies & Actionable Interventions
    print("\n" + "-" * 40)
    print(" TASK 4: INTERVENTION-ORIENTED SIMULATION")
    print("-" * 40)
    sample_indices = [0, 1]  # Pick two representative test students
    for i, idx in enumerate(sample_indices):
        student = X_test.iloc[[idx]]
        actual_class = "AT-RISK" if y_class_test.iloc[idx] == 1 else "SAFE / PASSING"
        actual_grade = y_reg_test.iloc[idx]
        
        prediction = model.predict_student(student)
        
        print(f"\n[Student Case #{i+1}] Actual: {actual_class} (Final Grade: {actual_grade}/20)")
        print(f" -> Predicted Status : {prediction['At-Risk Status']} (Risk Prob: {prediction['At-Risk Probability']})")
        print(f" -> Predicted Grade  : {prediction['Predicted Final Grade (0-20)']}/20")
        print(" -> Recommended Actionable Interventions:")
        for action in prediction["Recommended Interventions"]:
            print(f"    - {action}")

    print("\n" + "=" * 75)
    print(" Pipeline execution finished successfully.")
    print("=" * 75)


if __name__ == "__main__":
    # Default execution on Math dataset with early term (G1) indicator
    run_pipeline(subject="mat", include_g1=True)
