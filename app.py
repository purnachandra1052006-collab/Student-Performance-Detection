"""
Streamlit Web Application for Early Student Performance Prediction & Intervention
Run via: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_dataset
from feature_engineering import engineer_features, prepare_train_test_data
from model import EarlyStudentPerformanceModel

st.set_page_config(page_title="Early Student Performance Prediction", page_icon="🎓", layout="wide")


@st.cache_resource
def train_and_cache_model(subject: str = "mat"):
    raw_df = load_dataset(subject)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, feature_names = prepare_train_test_data(
        raw_df, include_g1=True, random_state=42
    )
    model = EarlyStudentPerformanceModel(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_class_train, y_reg_train)
    c_metrics = model.evaluate_classification(X_test, y_class_test)
    r_metrics = model.evaluate_regression(X_test, y_reg_test)
    return model, X_train, c_metrics, r_metrics, raw_df


def main():
    st.title("🎓 Early Student Performance Prediction & Intervention System")
    st.markdown(
        "Predict student academic outcomes early using demographic, academic, and behavioral features. "
        "Powered by a **Random Forest** Ensemble."
    )

    # Subject Selector
    subject_choice = st.sidebar.selectbox("Select Subject", ["Mathematics (mat)", "Portuguese (por)"], index=0)
    sub_code = "mat" if "mat" in subject_choice else "por"

    model, X_train, c_metrics, r_metrics, raw_df = train_and_cache_model(sub_code)

    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classification Accuracy", f"{c_metrics['Accuracy']:.1%}")
    m2.metric("At-Risk ROC-AUC", f"{c_metrics['ROC-AUC']:.3f}")
    m3.metric("Grade Prediction MAE", f"{r_metrics['MAE (Mean Absolute Error)']:.2f} / 20")
    m4.metric("Model R² Score", f"{r_metrics['R2 Score']:.3f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Student Profile & Early Indicators")
        
        c1, c2 = st.columns(2)
        with c1:
            g1 = st.slider("Early Term Grade (G1: 0-20)", 0, 20, 10, help="Period 1 exam score")
            failures = st.selectbox("Past Class Failures", [0, 1, 2, 3, 4], index=0)
            absences = st.slider("Number of Absences", 0, 50, 4)
            studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4], format_func=lambda x: {
                1: "< 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "> 10 hours"
            }[x], index=1)
            schoolsup = st.selectbox("Extra Educational Support (School)", ["no", "yes"])
            famsup = st.selectbox("Family Educational Support", ["yes", "no"])
            higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        
        with c2:
            freetime = st.slider("Free Time After School (1: very low - 5: very high)", 1, 5, 3)
            goout = st.slider("Going Out with Friends (1: very low - 5: very high)", 1, 5, 3)
            dalc = st.slider("Workday Alcohol (1: very low - 5: very high)", 1, 5, 1)
            walc = st.slider("Weekend Alcohol (1: very low - 5: very high)", 1, 5, 1)
            medu = st.slider("Mother's Education (0: none - 4: higher)", 0, 4, 2)
            fedu = st.slider("Father's Education (0: none - 4: higher)", 0, 4, 2)
            health = st.slider("Current Health Status (1: very bad - 5: very good)", 1, 5, 4)

    # Build student dataframe matching model schema
    sample_raw = raw_df.iloc[0:1].copy()
    sample_raw["G1"] = g1
    sample_raw["failures"] = failures
    sample_raw["absences"] = absences
    sample_raw["studytime"] = studytime
    sample_raw["freetime"] = freetime
    sample_raw["goout"] = goout
    sample_raw["Dalc"] = dalc
    sample_raw["Walc"] = walc
    sample_raw["Medu"] = medu
    sample_raw["Fedu"] = fedu
    sample_raw["health"] = health
    sample_raw["schoolsup"] = schoolsup
    sample_raw["famsup"] = famsup
    sample_raw["higher"] = higher

    # Re-engineer features for single record
    full_df = pd.concat([raw_df, sample_raw], ignore_index=True)
    X_full, _, _ = engineer_features(full_df, include_g1=True)
    student_vector = X_full.iloc[[-1]]

    # Ensure columns match train exactly
    student_vector = student_vector.reindex(columns=X_train.columns, fill_value=0)

    # Model Prediction
    prediction = model.predict_student(student_vector)

    with col_right:
        st.subheader("🎯 Prediction & Intervention Advisory")
        
        is_at_risk = "AT-RISK" in prediction["At-Risk Status"]
        if is_at_risk:
            st.error(f"### ⚠️ Status: {prediction['At-Risk Status']}")
        else:
            st.success(f"### ✅ Status: {prediction['At-Risk Status']}")

        p1, p2 = st.columns(2)
        p1.metric("Risk Probability", prediction["At-Risk Probability"])
        p2.metric("Predicted Final Grade (G3)", f"{prediction['Predicted Final Grade (0-20)']:.1f} / 20")

        st.progress(max(0.0, min(1.0, prediction["Predicted Final Grade (0-20)"] / 20.0)))

        st.subheader("💡 Prescriptive Interventions")
        for item in prediction["Recommended Interventions"]:
            st.info(f"👉 {item}")

    st.markdown("---")
    st.subheader("📊 Key Model Explainability Factors")
    top_df = model.get_feature_importances(top_n=8)
    st.dataframe(top_df[["Feature", "Combined_Importance", "Classification_Importance", "Regression_Importance"]], use_container_width=True)


if __name__ == "__main__":
    main()
