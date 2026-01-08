import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ------------------- LOAD MODEL ------------------- #
def load_model():
    models = {
        'Logistic Regression': joblib.load('logistic_regression_model.joblib'),
        'XGBoost': joblib.load('xgboost_model.joblib'),
        'SVM': joblib.load('svm_model.joblib')
    }
    feature_columns = joblib.load('feature_columns.joblib')
    scaler = joblib.load('feature_scaler.joblib')
    return models, feature_columns, scaler

# ------------------- PREPROCESS ------------------- #
def preprocess_input(input_data, feature_columns, scaler):
    processed_input = {}

    for col in feature_columns:
        if col.startswith('A') and col.endswith('_Score'):
            processed_input[col] = input_data.get(col, 0)
        elif col == 'age':
            processed_input[col] = np.log(float(input_data.get('age', 25.0)) + 1)
        elif col in ['gender', 'jaundice', 'austim', 'used_app_before', 'ethnicity',
                     'contry_of_res', 'relation', 'ageGroup']:
            processed_input[col] = input_data.get(col, 'unknown')
        elif col == 'result':
            processed_input[col] = float(input_data.get('result', 0.0))
        elif col in ['sum_score', 'ind']:
            processed_input[col] = int(input_data.get(col, 0))
        else:
            processed_input[col] = 0

    df = pd.DataFrame([processed_input])
    df = df[feature_columns]

    # Encode
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    df_scaled = scaler.transform(df_imputed)
    return df_scaled


# ------------------- MAIN UI ------------------- #
def main():
    st.set_page_config(page_title="ASD Prediction", layout="wide")

    # Beautiful header
    st.markdown(
        """
        <h1 style="text-align:center; color:#333; font-family: 'Arial Black';">
            Autism Spectrum Disorder Prediction
        </h1>
        <p style="text-align:center; font-size:18px; color:gray;">
            Enter the details below to evaluate ASD probability
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.title("⚙️ Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose a Model",
        ["Logistic Regression", "XGBoost", "SVM"]
    )

    models, feature_columns, scaler = load_model()
    input_data = {}

    st.write("")
    st.write("")

    # Card layout
    with st.container():
        st.markdown(
            """
            <div style="background:white; padding:25px; border-radius:12px; 
                        box-shadow:0 4px 15px rgba(0,0,0,0.1);">
            """,
            unsafe_allow_html=True
        )

        # Two-column layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧮 Basic Information")
            input_data['age'] = st.number_input("Age", 0.0, 100.0, 25.0)
            input_data['result'] = st.number_input("Result", -5.0, 100.0, 0.0)
            for i in range(1, 6):
                input_data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

        with col2:
            st.subheader("📌 Additional Details")
            input_data['gender'] = st.selectbox("Gender", ["male", "female"])
            input_data['jaundice'] = st.selectbox("Jaundice", ["yes", "no"])
            input_data['austim'] = st.selectbox("Autism", ["yes", "no"])
            input_data['used_app_before'] = st.selectbox("Used App Before", ["yes", "no"])
            for i in range(6, 11):
                input_data[f"A{i}_Score"] = st.number_input(f"A{i} Score", 0, 10, 0)

        # Feature engineering
        input_data['sum_score'] = sum(input_data[f"A{i}_Score"] for i in range(1, 11))
        input_data['ind'] = (
            (1 if input_data['austim'] == 'yes' else 0) +
            (1 if input_data['used_app_before'] == 'yes' else 0) +
            (1 if input_data['jaundice'] == 'yes' else 0)
        )

        def convertAge(age):
            if age < 4:
                return 'Toddler'
            elif age < 12:
                return 'Kid'
            elif age < 18:
                return 'Teenager'
            elif age < 40:
                return 'Young'
            else:
                return 'Senior'

        input_data['ageGroup'] = convertAge(input_data['age'])

        # Prediction Button
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    if st.button("🔍 Predict", use_container_width=True):
        processed_input = preprocess_input(input_data, feature_columns, scaler)

        model = models[model_choice]
        prediction = model.predict(processed_input)
        proba = model.predict_proba(processed_input)

        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ Potential Autism Spectrum Disorder Detected")
        else:
            st.success("✅ No Autism Spectrum Disorder Detected")

        st.info(f"**Probability of ASD:** {proba[0][1] * 100:.2f}%")
        st.write(f"Model Used: **{model_choice}**")


if __name__ == "__main__":
    main()
