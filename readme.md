# Autism Spectrum Disorder Prediction Model

## Project Overview

This project develops a machine learning solution for early detection of Autism Spectrum Disorder (ASD) using multiple classification algorithms. By leveraging advanced data preprocessing and machine learning techniques, the model aims to provide a reliable screening tool for potential ASD identification.

## Key Features

- Multiple Machine Learning Models:
  - Logistic Regression
  - XGBoost Classifier
  - Support Vector Machine (SVM)

- Advanced Data Preprocessing:
  - Feature engineering
  - Age group categorization
  - Log transformation
  - Label encoding
  - Feature scaling

- Handling Class Imbalance:
  - Random Over-Sampling technique to balance dataset

- Interactive Streamlit Web Application:
  - User-friendly interface
  - Model selection
  - Real-time ASD prediction


## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/autism-prediction.git
cd autism-prediction
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python train.py
```

### Running Streamlit App
```bash
streamlit run app.py
```

## Model Performance Metrics

- Logistic Regression:
  - Training AUC: ~0.85
  - Validation AUC: ~0.80

- XGBoost:
  - Training AUC: ~0.90
  - Validation AUC: ~0.85

- Support Vector Machine:
  - Training AUC: ~0.88
  - Validation AUC: ~0.82

## Dataset

The dataset includes various features related to:
- Demographic information
- Behavioral scores
- Medical history

## Preprocessing Techniques

- Log transformation of age
- Feature engineering
- Label encoding
- Standard scaling
- Handling class imbalance



