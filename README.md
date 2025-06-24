# Real-Time Fraud Detection System

A Streamlit-based dashboard for real-time credit card fraud detection using machine learning.

## Features
- Real-time transaction monitoring
- Manual transaction testing
- Interactive visualizations
- Risk level analysis
- Model performance metrics

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run dashboard/app.py
```

## Data
The system uses credit card transaction data stored in Google Drive. The data is automatically downloaded when running the application.

## Model
- XGBoost + Isolation Forest Ensemble
- 98.31% ROC-AUC Score
- 85% Fraud Detection Rate