# Real-Time Fraud Detection System

This is an interactive web application built with Streamlit that enables users to detect fraudulent credit card transactions using machine learning models. Designed with a user-friendly dashboard, the system provides real-time risk analysis, model performance metrics, and tools for manual transaction testing.

Leveraging a hybrid model of XGBoost and Isolation Forest, the app offers high accuracy and robust fraud detection, making it ideal for educational demos, proof-of-concept projects, or showcasing ML integration in fintech.

## View Demo

Watch the demo video to see the app in action:
you can find the video under the `frauddetectionsystemdemo.mp4` file in this repository.


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
