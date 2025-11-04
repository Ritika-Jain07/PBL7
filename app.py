# ======================================
# 🩺 Diabetes Prediction App + Gemini AI
# ======================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import google.generativeai as genai

# -------------------------------
# Load model, scaler, and features
# -------------------------------
best_model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

# -------------------------------
# Streamlit UI setup
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Sidebar for Gemini API Key
st.sidebar.header("🤖 Gemini AI Configuration")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details below to predict the probability of Diabetes.")

# -------------------------------
# Input fields
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    gender_val = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    urea = st.number_input("Urea (mg/dL)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    cr = st.number_input("Creatinine (Cr, mg/dL)", min_value=0.0, max_value=60.0, value=1.0, step=0.01)

with col2:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=400.0, value=180.0, step=1.0)
    tg = st.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
    hdl = st.number_input("HDL (mg/dL)", min_value=0.0, max_value=150.0, value=50.0, step=1.0)
    ldl = st.number_input("LDL (mg/dL)", min_value=0.0, max_value=200.0, value=100.0, step=1.0)

with col3:
    vldl = st.number_input("VLDL (mg/dL)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=60.0, value=22.0, step=0.1)

# -------------------------------
# Prediction button
# -------------------------------
if st.button("🔍 Predict Diabetes Risk"):
    # Prepare input
    gender_num = 1 if gender_val == "Male" else 0
    input_df = pd.DataFrame([{
        "Gender": gender_num,
        "AGE": age,
        "Urea": urea,
        "Cr": cr,
        "Chol": chol,
        "TG": tg,
        "HDL": hdl,
        "LDL": ldl,
        "VLDL": vldl,
        "BMI": bmi
    }])
    input_df = input_df[feature_order]

    input_scaled = scaler.transform(input_df)
    prob = best_model.predict_proba(input_scaled)[0][1]
    pred_class = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

    # -------------------------------
    # Display results
    # -------------------------------
    st.subheader("🧾 Prediction Result")
    st.write(f"**Predicted Probability of Diabetes:** {prob*100:.2f}%")
    st.write(f"**Predicted Class:** {pred_class}")

    # -------------------------------
    # LIME Explanation
    # -------------------------------
    st.subheader("🔍 LIME Explanation (Top Features)")

    lime_explainer = LimeTabularExplainer(
        training_data=np.array(scaler.transform(pd.DataFrame(np.zeros((1, len(feature_order))), columns=feature_order))),
        feature_names=feature_order,
        class_names=["Non-Diabetic", "Diabetic"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(input_scaled[0], best_model.predict_proba, num_features=6)
    lime_features = lime_exp.as_list()
    for feat, val in lime_features:
        st.write(f"• {feat}: {val:+.3f}")

    # -------------------------------
    # Medical Interpretation
    # -------------------------------
    st.subheader("🩸 Medical Interpretation")
    interpretations = {
        "BMI": "Higher BMI suggests overweight or obesity — ideal range is 18.5–24.9.",
        "Urea": "Elevated urea may signal kidney issues, common in diabetics.",
        "Cr": "High creatinine can indicate reduced kidney function.",
        "Chol": "High cholesterol increases heart disease risk in diabetics.",
        "TG": "High triglycerides are linked with insulin resistance.",
        "HDL": "Low HDL (good cholesterol) raises cardiovascular risk.",
        "LDL": "High LDL (bad cholesterol) worsens heart and diabetes outcomes.",
        "VLDL": "High VLDL indicates lipid metabolism issues.",
        "AGE": "Older age increases the likelihood of diabetes.",
        "Gender": "Males tend to have slightly higher risk due to fat distribution patterns."
    }

    interpreted_features = [feat.split()[0].replace('_', '') for feat, val in lime_features]
    for feat_name in interpreted_features:
        if feat_name in interpretations:
            st.write(f"• **{feat_name}**: {interpretations[feat_name]}")

    # -------------------------------
    # AI Recommendations (Gemini)
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI-Based Health & Lifestyle Recommendations")

    if st.button("💡 Generate Personalized Plan"):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar first.")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")

                prompt = f"""
                You are a medical AI assistant. Based on the following patient details:
                - Gender: {gender_val}
                - Age: {age}
                - BMI: {bmi}
                - Cholesterol: {chol}
                - HDL: {hdl}
                - LDL: {ldl}
                - Triglycerides: {tg}
                - Urea: {urea}
                - Creatinine: {cr}
                - Predicted Class: {pred_class}
                - Diabetes Probability: {prob*100:.2f}%

                Provide a personalized and easy-to-follow plan including:
                1. Daily diet plan (with Indian food examples)
                2. Exercise and lifestyle tips
                3. Health monitoring and preventive advice
                """

                with st.spinner("🧠 Generating your health plan..."):
                    response = model.generate_content(prompt)

                st.success("✅ Personalized health recommendations generated!")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"❌ Gemini API error: {e}")
                st.info("Please verify your API key and internet connection.")