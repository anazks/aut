import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
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

    df = df.replace({
        "yes": 1,
        "no": 0,
        "?": "Others",
        "others": "Others"
    })

    def convert_age(age):
        if age < 4:
            return "Toddler"
        elif age < 12:
            return "Child"
        elif age < 18:
            return "Adolescent"
        else:
            return "Adult"

    df["ageGroup"] = df["age"].apply(convert_age)
    df["sum_score"] = df.loc[:, "A1_Score":"A10_Score"].sum(axis=1)
    df["ind"] = df["austim"] + df["used_app_before"] + df["jaundice"]
    df["age_log"] = np.log1p(df["age"])

    return df

# ---------------- ENCODE CATEGORICALS ---------------- #
def encode_and_save(df):
    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col != "Class/ASD":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    joblib.dump(encoders, f"{WORK_DIR}/label_encoders.joblib")
    return df

# ---------------- PREPARE DATA ---------------- #
def prepare_data(df):
    drop_cols = ["ID", "age_desc", "used_app_before", "austim", "age"]

    X = df.drop(drop_cols + ["Class/ASD"], axis=1)
    y = df["Class/ASD"]

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

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

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
        score = recall_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t

# ---------------- TRAIN MODELS ---------------- #
def train_models(X_train, X_val, X_test, y_train, y_val, y_test):

    models = {
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE
        ),
        "LDA": LinearDiscriminantAnalysis()
    }

    results = []

    for name, model in models.items():
        print(f"\n🚀 Training {name}")
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, val_prob)

        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        metrics = {
            "Model": name,
            "AUC": roc_auc_score(y_test, test_prob),
            "Precision": precision_score(y_test, test_pred),
            "Recall": recall_score(y_test, test_pred),
            "F1": f1_score(y_test, test_pred),
            "MCC": matthews_corrcoef(y_test, test_pred),
            "Kappa": cohen_kappa_score(y_test, test_pred),
            "LogLoss": log_loss(y_test, test_prob)
        }

        results.append(metrics)

        print(classification_report(y_test, test_pred))
        for k, v in metrics.items():
            if k != "Model":
                print(f"{k}: {v:.4f}")

        ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
        plt.title(name)
        plt.show()

        joblib.dump(model, f"{WORK_DIR}/{name.lower()}_model.joblib")
        joblib.dump(threshold, f"{WORK_DIR}/{name.lower()}_threshold.joblib")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{WORK_DIR}/model_performance_metrics.csv", index=False)

    print("\n📊 Final Model Performance Summary:")
    print(results_df)

# ---------------- MAIN ---------------- #
def main():
    print("🧠 Training ASD Models (AdaBoost + LDA)")
    df = load_and_engineer_data("train.csv")
    df = encode_and_save(df)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)
    train_models(X_train, X_val, X_test, y_train, y_val, y_test)
    print("\n✅ Training complete. Artifacts saved to:", WORK_DIR)

if __name__ == "__main__":
    main()
