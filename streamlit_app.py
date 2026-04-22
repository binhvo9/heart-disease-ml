import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ===== Load model =====
@st.cache_resource
def load_model():
    return joblib.load("heart_rf_pipeline.pkl")

bundle = load_model()

# support cả 2 kiểu:
# 1) save cả dict {'model':..., ...}
# 2) save trực tiếp pipeline/model
if isinstance(bundle, dict) and "model" in bundle:
    model = bundle["model"]
    feature_columns = bundle.get("feature_columns", [])
    categorical_columns = bundle.get("categorical_columns", [])
else:
    model = bundle
    feature_columns = []

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Simple ML demo for portfolio")

st.markdown("### Enter patient data")

# ===== Inputs =====
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=130)
chol = st.number_input("Cholesterol", min_value=50, max_value=700, value=240)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
restecg = st.selectbox("Resting ECG", [0, 1, 2])
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal", [0, 1, 2, 3])

# map sex text -> model value
sex_value = 1 if sex == "Male" else 0

# ===== Build input row =====
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex_value,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "restecg": restecg,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

st.markdown("---")

# ===== Predict =====
if st.button("Predict Risk"):
    try:
        proba = model.predict_proba(input_data)[0][1]
        pred = int(proba >= 0.20)   # threshold m đang xài, sửa nếu muốn

        if proba < 0.20:
            risk_label = "Low Risk"
            risk_emoji = "🟢"
        elif proba < 0.50:
            risk_label = "Medium Risk"
            risk_emoji = "🟠"
        else:
            risk_label = "High Risk"
            risk_emoji = "🔴"

        st.subheader("Prediction Result")
        st.metric("Heart Disease Probability", f"{proba:.1%}")
        st.write(f"**Risk Level:** {risk_emoji} {risk_label}")
        st.write(f"**Predicted Class (threshold = 0.20):** {pred}")

        with st.expander("View input data"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")