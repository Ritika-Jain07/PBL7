import streamlit as st
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer
import google.generativeai as genai

# -------------------------------
# Gemini AI Setup (Hidden Branding)
# -------------------------------
genai.configure(api_key="AIzaSyD-GjQ66uumFjXbUGJZf4eENa4SXDQ4-vg")  # Replace with your Gemini 2.5 key
model_ai = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
# Load Model Files
# -------------------------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

st.set_page_config(page_title="Enhanced Diabetes Risk & Prediction System", layout="wide")

# Sidebar Phase Selector
st.sidebar.title("🩺 Enhanced Diabetes App")
phase = st.sidebar.radio(
    "Select Phase",
    ["Phase 1: Basic Screening", "Phase 2: Blood Test & Prediction", "Phase 3: Lifestyle Plan"]
)

st.title("🩺 Enhanced Diabetes Risk & Prediction System")
st.caption("Three Phases: **Basic Screening → Blood Test → AI Lifestyle Plan**")

# ==========================================================
# 🏥 Phase 1 — Basic Screening
# ==========================================================
if "Phase 1" in phase:
    st.subheader("🏥 Phase 1 — Basic Screening")

    col1, col2 = st.columns(2)
    with col1:
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=160.0)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=60.0)

    bmi = round(weight / ((height / 100) ** 2), 2)
    st.metric("Your BMI", bmi)

    if bmi < 18.5:
        st.info("Underweight — consider a nutrition consultation.")
    elif 18.5 <= bmi < 24.9:
        st.success("Normal weight — maintain a healthy lifestyle.")
    elif 25 <= bmi < 29.9:
        st.warning("Overweight — mild diabetes risk.")
    else:
        st.error("Obese — medical consultation recommended.")

    st.divider()
    st.markdown("### 🧍 Basic Lifestyle & Family History")

    family = st.radio("Family History of Diabetes?", ["Yes", "No"])
    activity = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low", "Sedentary"])
    diet = st.selectbox("Food Habit Type", ["Vegetarian", "Non-Vegetarian", "Mixed"])
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.session_state["age"] = age
    st.session_state["gender"] = gender
    st.session_state["bmi"] = bmi

    st.info("✔ Age, Gender, and BMI will be used later for AI insights.")

# ==========================================================
# 🧪 Phase 2 — Blood Test & Prediction
# ==========================================================
elif "Phase 2" in phase:
    st.subheader("🧪 Phase 2 — Blood Test Input & Prediction")

    col1, col2 = st.columns(2)
    with col1:
        urea = st.number_input("Urea (mg/dL)", min_value=0.0, max_value=200.0, value=30.0)
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=50.0, max_value=400.0, value=180.0)
        triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=20.0, max_value=500.0, value=150.0)
        hdl = st.number_input("HDL (mg/dL)", min_value=10.0, max_value=100.0, value=45.0)
    with col2:
        ldl = st.number_input("LDL (mg/dL)", min_value=10.0, max_value=300.0, value=110.0)
        vldl = st.number_input("VLDL (mg/dL)", min_value=5.0, max_value=100.0, value=25.0)
        fasting_glucose = st.number_input(
            "Fasting Glucose (mg/dL)",
            min_value=50.0, max_value=400.0, value=95.0,
            help="🩸 Normal: <100 | Pre-diabetic: 100–125 | Diabetic: ≥126"
        )
        post_glucose = st.number_input(
            "2-Hour Postprandial Glucose (mg/dL)",
            min_value=50.0, max_value=500.0, value=140.0,
            help="🩸 Normal: <140 | Pre-diabetic: 140–199 | Diabetic: ≥200"
        )
        hba1c = st.number_input(
            "HbA1c (%)",
            min_value=3.0, max_value=15.0, value=5.8,
            help="💉 Normal: <5.7 | Pre-diabetic: 5.7–6.4 | Diabetic: ≥6.5"
        )

    # 10 model features (as per your scaler)
    input_data = np.array([[urea, creatinine, cholesterol, triglycerides,
                            hdl, ldl, vldl, fasting_glucose,
                            post_glucose, hba1c]])

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("### 🧾 Model Prediction")
    st.write(f"**Predicted Probability of Diabetes:** {probability * 100:.2f}%")
    st.write(f"**Predicted Class (threshold 0.5):** {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")

    if hba1c >= 6.5 or fasting_glucose >= 126 or post_glucose >= 200:
        st.error("📋 Clinical Classification (ADA): 🔴 Diabetic")
    elif 5.7 <= hba1c < 6.5 or 100 <= fasting_glucose < 126:
        st.warning("📋 Clinical Classification (ADA): 🟠 Pre-Diabetic")
    else:
        st.success("📋 Clinical Classification (ADA): 🟢 Normal")

    st.markdown("### 🔍 Explainability (LIME)")
    explainer = LimeTabularExplainer(
        training_data=np.zeros((1, len(feature_order))),
        feature_names=feature_order,
        mode="classification"
    )
    exp = explainer.explain_instance(scaled_input[0], model.predict_proba, num_features=6)
    for f in exp.as_list():
        st.write(f"• {f[0]}: {f[1]:.3f}")

# ==========================================================
# 🌿 Phase 3 — Lifestyle & Diet Plan
# ==========================================================
elif "Phase 3" in phase:
    st.subheader("🌿 Phase 3 — Lifestyle & Diet Plan")

    st.write("Get a personalized **diet, exercise, and monitoring plan** based on your profile.")

    summary = f"Age: {st.session_state.get('age', 'N/A')}, Gender: {st.session_state.get('gender', 'N/A')}, BMI: {st.session_state.get('bmi', 'N/A')}"
    st.markdown(f"**Profile Summary:** {summary}")

    user_summary = st.text_area(
        "Enter any additional notes or preferences:",
        "Example: Pre-diabetic, vegetarian, sedentary lifestyle, wants Indian meal plan."
    )

    if st.button("Generate Personalized Plan"):
        prompt = f"""
        Create a practical daily lifestyle and diet plan for:
        {summary}. Additional notes: {user_summary}.
        Include:
        - Morning & evening routines
        - Sample Indian diet (Breakfast/Lunch/Dinner)
        - Exercise, hydration, sleep, and stress management tips
        - Short summary of how it helps diabetes prevention
        Do not mention any AI model name.
        """
        response = model_ai.generate_content(prompt)
        st.success("✅ Personalized Lifestyle Plan Generated:")
        st.markdown(response.text)
