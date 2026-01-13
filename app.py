import streamlit as st
import joblib
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

ARTIFACT_DIR = "model_artifacts"

# ---------------- PAGE STATE ---------------- #
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- LOAD ARTIFACTS ---------------- #
@st.cache_resource
def load_artifacts():
    models = {
        "AdaBoost": joblib.load(f"{ARTIFACT_DIR}/adaboost_model.joblib"),
        "LDA": joblib.load(f"{ARTIFACT_DIR}/lda_model.joblib"),
    }

    feature_columns = joblib.load(f"{ARTIFACT_DIR}/feature_columns.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/feature_scaler.joblib")
    imputer = joblib.load(f"{ARTIFACT_DIR}/imputer.joblib")
    encoders = joblib.load(f"{ARTIFACT_DIR}/label_encoders.joblib")

    return models, feature_columns, scaler, imputer, encoders


# ---------------- HELPERS ---------------- #
def yes_no(v):
    return 1 if v == "yes" else 0


def safe_encode(enc, v):
    v = str(v)
    return enc.transform([v])[0] if v in enc.classes_ else enc.transform([enc.classes_[0]])[0]


def age_group(age):
    if age < 4:
        return "Toddler"
    elif age < 12:
        return "Child"
    elif age < 18:
        return "Adolescent"
    else:
        return "Adult"


# ---------------- PREPROCESS ---------------- #
def preprocess_input(data, cols, scaler, imputer, encoders):
    row = {}

    for c in cols:
        if c.startswith("A"):
            row[c] = data[c]
        elif c == "age_log":
            row[c] = np.log1p(data["age"])
        elif c in ["gender", "jaundice", "austim", "used_app_before"]:
            row[c] = 1 if data[c] in ["yes", "male"] else 0
        elif c in ["sum_score", "ind"]:
            row[c] = data[c]
        elif c in encoders:
            row[c] = safe_encode(encoders[c], data.get(c))
        else:
            row[c] = 0

    df = pd.DataFrame([row])
    df = imputer.transform(df)
    df = scaler.transform(df)
    return df


# ---------------- PDF REPORT ---------------- #
def generate_pdf(prob, risk, age_grp, model_name):
    styles = getSampleStyleSheet()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(tmp.name)
    content = [
        Paragraph("<b>ASD Screening Report</b>", styles["Title"]),
        Paragraph(f"Age Group: {age_grp}", styles["Normal"]),
        Paragraph(f"Model Used: {model_name}", styles["Normal"]),
        Paragraph(f"Risk Level: <b>{risk}</b>", styles["Normal"]),
        Paragraph(f"ASD Probability: {prob:.2%}", styles["Normal"]),
        Paragraph(
            "<br/>This is a screening result generated using a machine learning model. "
            "It does not represent a medical diagnosis.",
            styles["Italic"]
        ),
    ]

    doc.build(content)
    return tmp.name


# ---------------- HOME SCREEN ---------------- #
def home_screen():
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(135deg, #4f46e5, #9333ea);
            padding: 60px;
            border-radius: 16px;
            color: white;
            text-align: center;
        }
        .card {
            background: white;
            padding: 25px;
            border-radius: 14px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="hero">
            <h1>🧠 Early Autism Screening System</h1>
            <p>AI-powered risk assessment based on clinical research</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>✅ Research-based models</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>👶 Child & Adult specific screening</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>📄 Downloadable PDF report</div>", unsafe_allow_html=True)

    st.write("")
    st.info(
        "This system follows published research. "
        "It is designed for screening purposes only."
    )

    st.write("")
    if st.button("🚀 Start Screening", use_container_width=True):
        st.session_state.page = "predict"
        st.rerun()


# ---------------- PREDICTION SCREEN ---------------- #
def prediction_screen():
    st.title("🔍 Autism Spectrum Disorder Screening")

    models, cols, scaler, imputer, encoders = load_artifacts()

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    left, right = st.columns(2)
    data = {}

    with left:
        data["age"] = st.number_input("Age", 0, 100, 6, 1)
        for i in range(1, 6):
            data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

    with right:
        data["gender"] = st.selectbox("Gender", ["male", "female"])
        data["jaundice"] = st.selectbox("Jaundice", ["yes", "no"])
        data["austim"] = st.selectbox("Family Autism History", ["yes", "no"])
        data["used_app_before"] = st.selectbox("Used App Before", ["yes", "no"])
        for i in range(6, 11):
            data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

    data["sum_score"] = sum(data[f"A{i}_Score"] for i in range(1, 11))
    data["ind"] = (
        yes_no(data["austim"])
        + yes_no(data["used_app_before"])
        + yes_no(data["jaundice"])
    )

    if st.button("🔍 Assess Risk", use_container_width=True):
        X = preprocess_input(data, cols, scaler, imputer, encoders)
        grp = age_group(data["age"])

        # Base-paper model selection
        model_name = "AdaBoost" if grp in ["Toddler", "Child"] else "LDA"
        prob = models[model_name].predict_proba(X)[0][1]

        if prob >= 0.65:
            risk, color = "HIGH RISK", "🔴"
        elif prob >= 0.45:
            risk, color = "MODERATE RISK", "🟠"
        else:
            risk, color = "LOW RISK", "🟢"

        st.subheader("📊 Screening Result")
        st.markdown(f"## {color} {risk}")
        st.progress(min(prob, 1.0))
        st.metric("ASD Probability", f"{prob:.2%}")
        st.write(f"**Age Group:** {grp}")
        st.write(f"**Model Used:** {model_name}")

        pdf_path = generate_pdf(prob, risk, grp, model_name)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="asd_screening_report.pdf"
            )


# ---------------- ROUTER ---------------- #
def main():
    st.set_page_config("Autism Screening", layout="wide")
    if st.session_state.page == "home":
        home_screen()
    else:
        prediction_screen()


if __name__ == "__main__":
    main()
