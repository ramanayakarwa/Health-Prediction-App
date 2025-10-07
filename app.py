import streamlit as st
import pickle
import pandas as pd
from fpdf import FPDF  # works with fpdf2

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Health Prediction Portal", layout="wide")
st.title("🩺 Comprehensive Health Prediction Portal")

# ------------------ Load Models ------------------
heart_rf, heart_features = pickle.load(open("models/heart_rf_model.pkl", "rb"))
diabetes_rf, diabetes_features = pickle.load(open("models/diabetes_rf_model.pkl", "rb"))
stroke_rf, stroke_features = pickle.load(open("models/stroke_rf_model.pkl", "rb"))

# ------------------ Tabs ------------------
tab1, tab2, tab3 = st.tabs(["❤️ Heart Disease", "🩸 Diabetes", "🧠 Stroke"])

# ------------------ Risk Categorization ------------------
def risk_category(prob):
    if prob < 0.3:
        return "Low"
    elif prob < 0.7:
        return "Moderate"
    else:
        return "High"

# ------------------ Risk Advice ------------------
risk_advice = {
    "Heart Disease": {
        "Low": "Maintain a healthy lifestyle and regular checkups.",
        "Moderate": "Monitor diet, exercise regularly, and consult a doctor.",
        "High": "Consult a cardiologist immediately and follow medical advice."
    },
    "Diabetes": {
        "Low": "Continue healthy eating habits and regular exercise.",
        "Moderate": "Monitor blood sugar levels and consult a doctor if needed.",
        "High": "Seek medical advice promptly and follow a treatment plan."
    },
    "Stroke": {
        "Low": "Maintain a healthy lifestyle and regular health checkups.",
        "Moderate": "Monitor blood pressure and cholesterol; consult your doctor.",
        "High": "Immediate medical attention recommended; follow preventive measures."
    }
}

# ------------------ Prediction Function ------------------
def predict_disease(model, user_input, all_features):
    df = pd.DataFrame([user_input])
    for f in all_features:
        if f not in df.columns:
            df[f] = 0
    df = df[all_features]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    prob = float(model.predict_proba(df)[0][1])
    category = risk_category(prob)
    return prob, category

# ------------------ Input Validation ------------------
def validate_inputs(user_input):
    if not user_input or all(v == "" for v in user_input.values()):
        return "no_data"
    if any(v is None or v == "" for v in user_input.values()):
        return "empty"
    if all(float(v) == 0 for v in user_input.values()):
        return "all_zero"
    return "ok"

# ------------------ Initialize Session State ------------------
if "results" not in st.session_state:
    st.session_state["results"] = {
        "Heart Disease": (None, None),
        "Diabetes": (None, None),
        "Stroke": (None, None)
    }

# ------------------ HEART DISEASE TAB ------------------
with tab1:
    st.subheader("❤️ Heart Disease Prediction")
    heart_input_features = [col for col in heart_features if col not in ["Gender_M"]]
    user_input = {}
    for col in heart_input_features:
        user_input[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, key=f"heart_{col}")

    if st.button("Predict Heart Disease"):
        status = validate_inputs(user_input)
        if status == "no_data":
            st.warning("⚠️ No patient data entered, so no prediction.")
        elif status == "empty":
            st.warning("⚠️ Please fill all fields before prediction.")
        elif status == "all_zero":
            st.warning("⚠️ Please enter at least one non-zero value.")
        else:
            heart_prob, heart_category = predict_disease(heart_rf, user_input, heart_features)
            st.session_state["results"]["Heart Disease"] = (heart_prob, heart_category)
            color = 'red' if heart_category == "High" else 'orange' if heart_category == "Moderate" else 'green'
            st.markdown(f"<h3 style='color:{color};'>Risk: {heart_prob*100:.2f}% ({heart_category})</h3>", unsafe_allow_html=True)
            st.info(risk_advice["Heart Disease"][heart_category])

# ------------------ DIABETES TAB ------------------
with tab2:
    st.subheader("🩸 Diabetes Prediction")
    gender_col = next((col for col in diabetes_features if col.lower().startswith("gender_")), None)
    diabetes_input_features = [col for col in diabetes_features if col != gender_col]
    user_input = {}
    if gender_col:
        gender_option = st.selectbox("Gender", ["Male", "Female"], key="diabetes_gender")
        user_input[gender_col] = 1 if gender_option == "Male" else 0
    for col in diabetes_input_features:
        user_input[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, key=f"diabetes_{col}")

    if st.button("Predict Diabetes"):
        status = validate_inputs(user_input)
        if status == "no_data":
            st.warning("⚠️ No patient data entered, so no prediction.")
        elif status == "empty":
            st.warning("⚠️ Please fill all fields before prediction.")
        elif status == "all_zero":
            st.warning("⚠️ Please enter at least one non-zero value.")
        else:
            diabetes_prob, diabetes_category = predict_disease(diabetes_rf, user_input, diabetes_features)
            st.session_state["results"]["Diabetes"] = (diabetes_prob, diabetes_category)
            color = 'red' if diabetes_category == "High" else 'orange' if diabetes_category == "Moderate" else 'green'
            st.markdown(f"<h3 style='color:{color};'>Risk: {diabetes_prob*100:.2f}% ({diabetes_category})</h3>", unsafe_allow_html=True)
            st.info(risk_advice["Diabetes"][diabetes_category])

# ------------------ STROKE TAB ------------------
with tab3:
    st.subheader("🧠 Stroke Prediction")
    remove_cols = [col for col in stroke_features if "id" in col.lower() or "AGE" in col or "gender_M" in col]
    stroke_input_features = [col for col in stroke_features if col not in remove_cols]
    user_input = {}
    for col in stroke_input_features:
        user_input[col] = st.number_input(col.replace("_", " ").title(), min_value=0.0, key=f"stroke_{col}")

    if st.button("Predict Stroke"):
        status = validate_inputs(user_input)
        if status == "no_data":
            st.warning("⚠️ No patient data entered, so no prediction.")
        elif status == "empty":
            st.warning("⚠️ Please fill all fields before prediction.")
        elif status == "all_zero":
            st.warning("⚠️ Please enter at least one non-zero value.")
        else:
            stroke_prob, stroke_category = predict_disease(stroke_rf, user_input, stroke_features)
            st.session_state["results"]["Stroke"] = (stroke_prob, stroke_category)
            color = 'red' if stroke_category == "High" else 'orange' if stroke_category == "Moderate" else 'green'
            st.markdown(f"<h3 style='color:{color};'>Risk: {stroke_prob*100:.2f}% ({stroke_category})</h3>", unsafe_allow_html=True)
            st.info(risk_advice["Stroke"][stroke_category])

# ------------------ PDF GENERATION ------------------
def generate_pdf(results):
    pdf = FPDF()
    pdf.add_page()

    # Add DejaVuSans regular for Unicode support
    pdf.add_font('DejaVu', '', 'fonts/dejavu-sans-ttf-2.37/ttf/DejaVuSans.ttf', uni=True)
    pdf.set_font('DejaVu', '', 16)
    pdf.cell(0, 10, "Health Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font('DejaVu', '', 12)
    for disease, (prob, category) in results.items():
        if prob is not None:
            pdf.multi_cell(0, 8,
                           f"{disease}:\n"
                           f"  - Risk Probability: {prob*100:.2f}%\n"
                           f"  - Risk Category: {category}\n"
                           f"  - Advice: {risk_advice[disease][category]}\n")
            pdf.ln(2)

    pdf.set_font('DejaVu', '', 10)
    pdf.multi_cell(0, 6,
                   "Disclaimer: This report is for informational purposes only and does not replace professional medical advice.",
                   align="L")

    pdf_file = "Health_Report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# ------------------ DOWNLOAD PDF ------------------
if st.button("Download Health Report"):
    results = st.session_state["results"]
    if any(v[0] is not None for v in results.values()):
        pdf_file = generate_pdf(results)
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download PDF", data=f, file_name="Health_Report.pdf", mime="application/pdf")
    else:
        st.warning("⚠️ Please make at least one prediction before downloading the report.")
