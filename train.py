import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.impute import SimpleImputer
import joblib
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Ensure working directory is set to Kaggle's working directory
working_dir = '/kaggle/working'
os.makedirs(working_dir, exist_ok=True)

def load_and_preprocess_data(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Replace values
    df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})
    
    # Age conversion function
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
    
    # Feature engineering
    df['ageGroup'] = df['age'].apply(convertAge)
    
    def add_feature(data):
        # Creating a column with sum of scores
        data['sum_score'] = data.loc[:,'A1_Score':'A10_Score'].sum(axis=1)
        
        # Creating an indicator feature
        data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
        
        return data
    
    df = add_feature(df)
    
    # Log transformation of age
    df['age'] = np.log(df['age'])
    
    # Label encoding
    def encode_labels(data):
        for col in data.columns:
            if data[col].dtype == 'object':
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        return data
    
    df = encode_labels(df)
    
    return df

def prepare_data(df):
    # Remove unnecessary columns
    removal = ['ID', 'age_desc', 'used_app_before', 'austim']
    features = df.drop(removal + ['Class/ASD'], axis=1)
    target = df['Class/ASD']
    
    # Split the data
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    # Oversample minority class
    ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train_imputed, Y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    return X_scaled, Y_resampled, X_val_scaled, Y_val, scaler, X_train.columns

def train_and_save_models(X, Y, X_val, Y_val):
    # Define models
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'xgboost': XGBClassifier(),
        'svm': SVC(kernel='rbf', probability=True)
    }
    
    # Train and save models
    results = {}
    for name, model in models.items():
        # Train the model
        model.fit(X, Y)
        
        # Predict and calculate metrics
        train_pred = model.predict(X)
        val_pred = model.predict(X_val)
        
        # Store results
        results[name] = {
            'train_auc': metrics.roc_auc_score(Y, train_pred),
            'val_auc': metrics.roc_auc_score(Y_val, val_pred),
            'train_report': classification_report(Y, train_pred),
            'val_report': classification_report(Y_val, val_pred),
            'model': model
        }
        
        # Save the model
        joblib.dump(model, os.path.join(working_dir, f'{name}_model.joblib'))
    
    return results

def main():
    # Load and preprocess data
    train_path = 'train.csv'
    df = load_and_preprocess_data(train_path)
    
    # Prepare data
    X, Y, X_val, Y_val, scaler, feature_columns = prepare_data(df)
    
    # Save scaler, feature columns, and label encoder
    joblib.dump(scaler, os.path.join(working_dir, 'feature_scaler.joblib'))
    joblib.dump(feature_columns.tolist(), os.path.join(working_dir, 'feature_columns.joblib'))
    
    # Train and save models
    results = train_and_save_models(X, Y, X_val, Y_val)
    
    # Print results
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()} Model Results:")
        print(f"Training AUC: {model_results['train_auc']}")
        print(f"Validation AUC: {model_results['val_auc']}")
        print("\nTraining Classification Report:")
        print(model_results['train_report'])
        print("\nValidation Classification Report:")
        print(model_results['val_report'])
    
    # Visualize Confusion Matrix for Logistic Regression
    plt.figure(figsize=(8,6))
    ConfusionMatrixDisplay.from_estimator(results['logistic_regression']['model'], X_val, Y_val)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    main()