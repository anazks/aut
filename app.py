import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- CONFIG ---------------- #
ARTIFACT_DIR = "model_artifacts"

# ---------------- LOAD ARTIFACTS ---------------- #
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load(f"{ARTIFACT_DIR}/logistic_regression_model.joblib"),
        "SVM": joblib.load(f"{ARTIFACT_DIR}/svm_model.joblib"),
        "Random Forest": joblib.load(f"{ARTIFACT_DIR}/random_forest_model.joblib"),
        "XGBoost": joblib.load(f"{ARTIFACT_DIR}/xgboost_model.joblib"),
        "LDA": joblib.load(f"{ARTIFACT_DIR}/lda_model.joblib"),  # ✅ ADDED
    }

    thresholds = {
        "Logistic Regression": joblib.load(f"{ARTIFACT_DIR}/logistic_regression_threshold.joblib"),
        "SVM": joblib.load(f"{ARTIFACT_DIR}/svm_threshold.joblib"),
        "Random Forest": joblib.load(f"{ARTIFACT_DIR}/random_forest_threshold.joblib"),
        "XGBoost": joblib.load(f"{ARTIFACT_DIR}/xgboost_threshold.joblib"),
        "LDA": joblib.load(f"{ARTIFACT_DIR}/lda_threshold.joblib"),  # ✅ ADDED
    }

    feature_columns = joblib.load(f"{ARTIFACT_DIR}/feature_columns.joblib")
    scaler = joblib.load(f"{ARTIFACT_DIR}/feature_scaler.joblib")
    imputer = joblib.load(f"{ARTIFACT_DIR}/imputer.joblib")
    encoders = joblib.load(f"{ARTIFACT_DIR}/label_encoders.joblib")

    return models, thresholds, feature_columns, scaler, imputer, encoders


# ---------------- HELPERS ---------------- #
def yes_no_to_int(v):
    return 1 if v == "yes" else 0


def safe_label_encode(encoder, value):
    value = str(value)
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([encoder.classes_[0]])[0]


# ---------------- PREPROCESS INPUT ---------------- #
def preprocess_input(input_data, feature_columns, scaler, imputer, encoders):
    row = {}

    for col in feature_columns:

        if col.startswith("A") and col.endswith("_Score"):
            row[col] = input_data[col]

        elif col == "age_log":
            row[col] = np.log1p(input_data["age"])

        elif col in ["gender", "jaundice", "austim", "used_app_before"]:
            if col == "gender":
                row[col] = 1 if input_data[col] == "male" else 0
            else:
                row[col] = yes_no_to_int(input_data[col])

        elif col == "sum_score":
            row[col] = input_data["sum_score"]

        elif col == "ind":
            row[col] = input_data["ind"]

        elif col in encoders:
            row[col] = safe_label_encode(encoders[col], input_data.get(col))

        else:
            row[col] = 0

    df = pd.DataFrame([row])
    df = imputer.transform(df)
    df = scaler.transform(df)

    return df


# ---------------- UI ---------------- #
def main():
    st.set_page_config("ASD Prediction", layout="wide")

    st.title("🧠 Autism Spectrum Disorder Screening")
    st.caption("Screening tool only — not a medical diagnosis.")

    models, thresholds, feature_columns, scaler, imputer, encoders = load_artifacts()

    st.sidebar.title("⚙️ Model Selection")
    model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))

    input_data = {}

    col1, col2 = st.columns(2)

    with col1:
        input_data["age"] = st.number_input("Age", 0, 100, 25, 1)
        for i in range(1, 6):
            input_data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

    with col2:
        input_data["gender"] = st.selectbox("Gender", ["male", "female"])
        input_data["jaundice"] = st.selectbox("Jaundice", ["yes", "no"])
        input_data["austim"] = st.selectbox("Autism", ["yes", "no"])
        input_data["used_app_before"] = st.selectbox("Used App Before", ["yes", "no"])
        for i in range(6, 11):
            input_data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

    input_data["sum_score"] = sum(input_data[f"A{i}_Score"] for i in range(1, 11))
    input_data["ind"] = (
        yes_no_to_int(input_data["austim"])
        + yes_no_to_int(input_data["used_app_before"])
        + yes_no_to_int(input_data["jaundice"])
    )

    if st.button("🔍 Predict", use_container_width=True):
        X = preprocess_input(input_data, feature_columns, scaler, imputer, encoders)

        model = models[model_choice]
        threshold = thresholds[model_choice]

        prob = model.predict_proba(X)[0][1]
        pred = int(prob >= threshold)

        st.subheader("📊 Prediction Result")

        if pred == 1:
            st.error("⚠️ Potential Autism Spectrum Disorder Detected")
        else:
            st.success("✅ No Autism Spectrum Disorder Detected")

        st.metric("ASD Probability", f"{prob * 100:.2f}%")
        st.caption(f"Decision Threshold: {threshold:.2f}")
        st.write(f"Model Used: **{model_choice}**")

        st.subheader("🔍 Model-wise Probabilities")
        for name, m in models.items():
            p = m.predict_proba(X)[0][1]
            st.write(f"{name}: {p * 100:.2f}%")


if __name__ == "__main__":
    main()
