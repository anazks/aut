import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

# ---------------- CONFIG ---------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

WORK_DIR = "model_artifacts"
os.makedirs(WORK_DIR, exist_ok=True)

# ---------------- LOAD & FEATURE ENGINEERING ---------------- #
def load_and_engineer_data(path):
    df = pd.read_csv(path)

    # Normalize binary values
    df = df.replace({
        'yes': 1,
        'no': 0,
        '?': 'Others',
        'others': 'Others'
    })

    def convert_age(age):
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

    df['ageGroup'] = df['age'].apply(convert_age)

    # Engineered features
    df['sum_score'] = df.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
    df['ind'] = df['austim'] + df['used_app_before'] + df['jaundice']

    # IMPORTANT: training age feature
    df['age_log'] = np.log1p(df['age'])

    return df

# ---------------- ENCODE CATEGORICALS ---------------- #
def encode_and_save(df):
    encoders = {}

    for col in df.select_dtypes(include='object').columns:
        if col != 'Class/ASD':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    joblib.dump(encoders, f"{WORK_DIR}/label_encoders.joblib")
    return df

# ---------------- PREPARE DATA ---------------- #
def prepare_data(df):
    drop_cols = ['ID', 'age_desc', 'used_app_before', 'austim', 'age']

    X = df.drop(drop_cols + ['Class/ASD'], axis=1)
    y = df['Class/ASD']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Oversample
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    # Scale (needed for LR, SVM, LDA)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save transformers
    joblib.dump(imputer, f"{WORK_DIR}/imputer.joblib")
    joblib.dump(scaler, f"{WORK_DIR}/feature_scaler.joblib")
    joblib.dump(list(X.columns), f"{WORK_DIR}/feature_columns.joblib")

    return X_train, X_val, X_test, y_train, y_val, y_test

# ---------------- THRESHOLD OPTIMIZATION ---------------- #
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_t, best_score = 0.5, 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=2)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t

# ---------------- TRAIN MODELS ---------------- #
def train_models(X_train, X_val, X_test, y_train, y_val, y_test):

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),

        "SVM": SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=RANDOM_STATE
        ),

        # 🔥 LINEAR DISCRIMINANT ANALYSIS
        "LDA": LinearDiscriminantAnalysis()
    }

    for name, model in models.items():
        print(f"\n🚀 Training {name}")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, val_prob)

        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        print(f"Optimal Threshold: {threshold:.2f}")
        print(classification_report(y_test, test_pred))
        print(f"AUC: {roc_auc_score(y_test, test_prob):.4f}")

        ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
        plt.title(name)
        plt.show()

        # Save model + threshold
        joblib.dump(
            model,
            f"{WORK_DIR}/{name.lower().replace(' ', '_')}_model.joblib"
        )
        joblib.dump(
            threshold,
            f"{WORK_DIR}/{name.lower().replace(' ', '_')}_threshold.joblib"
        )

# ---------------- MAIN ---------------- #
def main():
    print("🧠 Training ASD Models (LR + SVM + RF + XGB + LDA)")
    df = load_and_engineer_data("train.csv")
    df = encode_and_save(df)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
    train_models(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\n✅ Training complete. Artifacts saved to:", WORK_DIR)

if __name__ == "__main__":
    main()
