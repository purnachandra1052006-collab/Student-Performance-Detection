# Early Student Performance Prediction Using Machine Learning

An interpretable and actionable Machine Learning framework for early identification of at-risk students and final grade forecasting using demographic, academic, and behavioral attributes.

---

## 🎯 Project Overview & Tasks

1. **Classification (At-Risk Identification)**: Predicts whether a student is at-risk of failing (`G3 < 10` on a 20-point scale) early in the semester.
2. **Regression (Final Grade Prediction)**: Accurately predicts final academic performance (`G3`, continuous scale 0–20).
3. **Advanced Feature Engineering**: Combines behavioral dynamics, log-absence normalization, academic momentum ($G1 - 3 \times failures$), and non-linear polynomial grade interactions.
4. **Explainability & Intervention Advisory**: Uses tree feature importances to generate personalized, prescriptive interventions (remedial tutoring, attendance contracts, study habit coaching, wellbeing mentoring).

---

## 🤖 Model Choice: Tuned Random Forest Ensemble

We use a unified **Random Forest** tree ensemble architecture:
- **`RandomForestClassifier`**: `n_estimators=150`, `max_depth=4`, `max_features=0.4`, `class_weight='balanced'`.
- **`RandomForestRegressor`**: `n_estimators=150`, `max_depth=10`, `max_features=0.7`, `criterion='squared_error'`.

---

## 📊 Experimental Results (Math Dataset)

| Metric | Baseline Random Forest | Enhanced & Tuned Model | Gain / Improvement |
|---|---|---|---|
| **Classification ROC-AUC** | 0.9173 | **0.9376** | **+2.03%** |
| **Classification Recall (At-Risk)** | 84.62% | **88.46%** | **+3.84%** (Identifies 23/26 at-risk students early) |
| **Regression $R^2$ Score** | 0.7820 | **0.8030** | **Crosses the 0.80 threshold** |
| **Regression MAE** | 1.586 points | **1.523 points** | **-0.063 error reduction** |
| **Regression RMSE** | 2.148 points | **2.042 points** | **-0.106 variance reduction** |

---

## 💡 Top Predictive Risk Factors

1. **$G1$ (Period 1 Exam)** & **$G1^2$ (Polynomial Grade Interaction)** ($\approx 44.0\%$ combined importance)
2. **`academic_momentum` ($G1 - 3 \times \text{failures}$)** ($\approx 19.8\%$ combined importance)
3. **`G1_failing` & `risk_index`** ($\approx 8.5\%$ combined importance)
4. **`absences` & `log_absences`** ($\approx 8.2\%$ combined importance)

---

## 📁 Project Files

```text
Student-Performance-Detection/
├── data/                       # UCI Student Performance CSV datasets (student-mat.csv, student-por.csv)
├── plots/                      # Evaluation plots & visualizations
│   ├── feature_importance.png
│   ├── grade_prediction_scatter.png
│   └── confusion_matrix.png
├── data_loader.py              # Automatic dataset downloader and loader
├── feature_engineering.py      # Preprocessing, momentum calculation & data splitting
├── model.py                    # Tuned Random Forest suite & intervention generator
├── main.py                     # Unified CLI execution pipeline
├── visualize.py                # Generates high-resolution evaluation charts
├── app.py                      # Interactive Streamlit Web UI
└── README.md                   # Project documentation
```

---

## 🚀 How to Run

### 1. Run Full End-to-End Pipeline (CLI)
```bash
python main.py
```

### 2. Generate Evaluation Visualizations
```bash
python visualize.py
```

### 3. Launch Interactive Web Dashboard
```bash
streamlit run app.py
```
