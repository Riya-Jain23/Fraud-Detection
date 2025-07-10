import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import time
import gdown
from datetime import datetime, timedelta
import random
import logging

# Add parent directory to path to import our model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fraud_detector import FraudDetector

# Page config
st.set_page_config(
    page_title="Real-Time Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-low {
        background-color: #44ff44;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Logging configuration
logging.basicConfig(filename='transactions.log', level=logging.INFO)

@st.cache_resource
def load_fraud_detector():
    """Load the trained fraud detection model"""
    try:
        detector = FraudDetector()
        detector.load_models('../models/fraud_detection_models.pkl')
        return detector
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

import gdown

@st.cache_data
def load_sample_data():
    """Download and load sample data from Google Drive"""
    try:
        file_id = "1fws1m6q_jXA7r_lhwszbPTGocrKqfaXk"
        output_path = "creditcard.csv"

        # Only download if file is not already present
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)

        df = pd.read_csv(output_path)
        return df.sample(n=1000, random_state=42)  # Sample for simulation
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def generate_fake_transaction(base_data):
    """Generate a realistic fake transaction for simulation"""
    # Pick a random transaction as base
    base_transaction = base_data.sample(n=1).iloc[0]
    
    # Add some realistic variations
    fake_transaction = base_transaction.copy()
    
    # Vary the amount
    fake_transaction['Amount'] = max(1, base_transaction['Amount'] + random.uniform(-50, 200))
    
    # Vary some V features slightly
    for col in ['V1', 'V2', 'V3', 'V4']:
        if col in fake_transaction.index:
            fake_transaction[col] += random.uniform(-0.1, 0.1)
    
    # Current time
    fake_transaction['Time'] = time.time()
    
    return fake_transaction

def log_transaction(transaction, result):
    logging.info(f"Transaction: {transaction}, Result: {result}")

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Real-Time Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Load model and data
    detector = load_fraud_detector()
    sample_data = load_sample_data()
    
    if detector is None or sample_data is None:
        st.error("Failed to load model or data. Please check file paths.")
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Control Panel")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (Real-time)", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 3)
    
    # Manual transaction input
    st.sidebar.header("🔍 Test Single Transaction")
    
    # Option to generate a random transaction
    if st.sidebar.button("🎲 Generate Random Transaction"):
        st.session_state.test_transaction = generate_fake_transaction(sample_data)
    
    # Option to manually input transaction details
    st.sidebar.subheader("✏️ Enter Transaction Details")
    manual_transaction = {}
    manual_transaction['Time'] = st.sidebar.number_input("Time", value=406.000000)
    manual_transaction['V1'] = st.sidebar.number_input("V1", value=-2.312227)
    manual_transaction['V2'] = st.sidebar.number_input("V2", value=1.951992)
    manual_transaction['V3'] = st.sidebar.number_input("V3", value=-1.609851)
    manual_transaction['V4'] = st.sidebar.number_input("V4", value=3.997906)
    manual_transaction['V5'] = st.sidebar.number_input("V5", value=-0.522188)
    manual_transaction['V6'] = st.sidebar.number_input("V6", value=-1.426545)
    manual_transaction['V7'] = st.sidebar.number_input("V7", value=-2.537387)
    manual_transaction['V8'] = st.sidebar.number_input("V8", value=1.391657)
    manual_transaction['V9'] = st.sidebar.number_input("V9", value=-2.770089)
    manual_transaction['V10'] = st.sidebar.number_input("V10", value=-2.772272)
    manual_transaction['V11'] = st.sidebar.number_input("V11", value=3.202033)
    manual_transaction['V12'] = st.sidebar.number_input("V12", value=-2.899907)
    manual_transaction['V13'] = st.sidebar.number_input("V13", value=-0.595222)
    manual_transaction['V14'] = st.sidebar.number_input("V14", value=-4.289254)
    manual_transaction['V15'] = st.sidebar.number_input("V15", value=0.389724)
    manual_transaction['V16'] = st.sidebar.number_input("V16", value=-1.140747)
    manual_transaction['V17'] = st.sidebar.number_input("V17", value=-2.830056)
    manual_transaction['V18'] = st.sidebar.number_input("V18", value=-0.016822)
    manual_transaction['V19'] = st.sidebar.number_input("V19", value=0.416956)
    manual_transaction['V20'] = st.sidebar.number_input("V20", value=0.126911)
    manual_transaction['V21'] = st.sidebar.number_input("V21", value=0.517232)
    manual_transaction['V22'] = st.sidebar.number_input("V22", value=-0.035049)
    manual_transaction['V23'] = st.sidebar.number_input("V23", value=-0.465211)
    manual_transaction['V24'] = st.sidebar.number_input("V24", value=0.320198)
    manual_transaction['V25'] = st.sidebar.number_input("V25", value=0.044519)
    manual_transaction['V26'] = st.sidebar.number_input("V26", value=0.177840)
    manual_transaction['V27'] = st.sidebar.number_input("V27", value=0.261145)
    manual_transaction['V28'] = st.sidebar.number_input("V28", value=-0.143276)
    manual_transaction['Amount'] = st.sidebar.number_input("Amount", value=0.000000)
    
    if st.sidebar.button("💡 Test Manual Transaction"):
        st.session_state.test_transaction = manual_transaction
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize session state for metrics
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
        st.session_state.fraud_detected = 0
        st.session_state.high_risk = 0
        st.session_state.recent_transactions = []
    
    # Simulate real-time processing
    if auto_refresh:
        # Generate new transaction
        new_transaction = generate_fake_transaction(sample_data)
        
        # Remove 'Class' if it exists (for prediction)
        transaction_features = new_transaction.drop('Class', errors='ignore')
        
        # Predict fraud
        result = detector.predict_single_transaction(transaction_features.to_dict())
        
        # Log the transaction and result
        log_transaction(transaction_features.to_dict(), result)
        
        # Update metrics
        st.session_state.total_processed += 1
        if result['is_fraud']:
            st.session_state.fraud_detected += 1
        if result['risk_level'] == 'HIGH':
            st.session_state.high_risk += 1
        
        # Store recent transaction
        transaction_record = {
            'timestamp': datetime.now(),
            'amount': new_transaction['Amount'],
            'fraud_prob': result['fraud_probability'],
            'risk_level': result['risk_level'],
            'is_fraud': result['is_fraud']
        }
        
        st.session_state.recent_transactions.append(transaction_record)
        if len(st.session_state.recent_transactions) > 50:
            st.session_state.recent_transactions.pop(0)
    
    # Display metrics
    with col1:
        st.metric(
            label="📊 Total Processed",
            value=st.session_state.total_processed,
            delta=1 if auto_refresh else 0
        )
    
    with col2:
        fraud_rate = (st.session_state.fraud_detected / max(st.session_state.total_processed, 1)) * 100
        st.metric(
            label="🚨 Fraud Detected",
            value=st.session_state.fraud_detected,
            delta=f"{fraud_rate:.2f}%"
        )
    
    with col3:
        st.metric(
            label="⚠️ High Risk",
            value=st.session_state.high_risk,
            delta="Critical" if st.session_state.high_risk > 5 else "Normal"
        )
    
    with col4:
        accuracy = 98.31  # From our model evaluation
        st.metric(
            label="🎯 Model Accuracy",
            value=f"{accuracy}%",
            delta="Excellent"
        )
    
    # Real-time transaction feed
    st.header("📈 Live Transaction Feed")
    
    if st.session_state.recent_transactions:
        # Create DataFrame from recent transactions
        df_recent = pd.DataFrame(st.session_state.recent_transactions)
        
        # Display recent transactions table
        st.subheader("🔍 Recent Transactions")
        
        # Color code by risk level
        def color_risk(val):
            if val == 'HIGH':
                return 'background-color: #ff4444; color: white'
            elif val == 'MEDIUM':
                return 'background-color: #ffaa00; color: white'
            else:
                return 'background-color: #44ff44; color: black'
        
        styled_df = df_recent.tail(10).style.map(color_risk, subset=['risk_level'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Real-time charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fraud probability over time
            fig_prob = px.line(
                df_recent.tail(20), 
                x='timestamp', 
                y='fraud_prob',
                title='🎯 Fraud Probability Over Time',
                color_discrete_sequence=['#1f77b4']
            )
            fig_prob.add_hline(y=0.5, line_dash="dash", line_color="red", 
                              annotation_text="Fraud Threshold")
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Transaction amounts
            fig_amount = px.scatter(
                df_recent.tail(20),
                x='timestamp',
                y='amount',
                color='risk_level',
                size='fraud_prob',
                title='💰 Transaction Amounts by Risk Level',
                color_discrete_map={
                    'LOW': '#44ff44',
                    'MEDIUM': '#ffaa00', 
                    'HIGH': '#ff4444'
                }
            )
            st.plotly_chart(fig_amount, use_container_width=True)
        
        # Risk level distribution
        risk_counts = df_recent['risk_level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='🔍 Risk Level Distribution',
            color_discrete_map={
                'LOW': '#44ff44',
                'MEDIUM': '#ffaa00',
                'HIGH': '#ff4444'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Manual transaction testing
    st.header("🧪 Manual Transaction Testing")
    
    if 'test_transaction' in st.session_state:
        st.subheader("Test Transaction Details")
        
        # Display transaction details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Transaction Features:**")
            st.write(f"Amount: ${st.session_state.test_transaction['Amount']:.2f}")
            st.write(f"Time: {st.session_state.test_transaction['Time']}")
            
        with col2:
            # Predict fraud for test transaction
            test_features = st.session_state.test_transaction  # No need to drop anything
            test_result = detector.predict_single_transaction(test_features)
            
            st.write("**Fraud Analysis:**")
            st.write(f"Fraud Probability: {test_result['fraud_probability']:.4f}")
            st.write(f"Risk Level: {test_result['risk_level']}")
            st.write(f"Predicted as Fraud: {test_result['is_fraud']}")
            
            # Alert styling
            if test_result['risk_level'] == 'HIGH':
                st.markdown(f'<div class="alert-high">🚨 HIGH RISK TRANSACTION DETECTED!</div>', 
                           unsafe_allow_html=True)
            elif test_result['risk_level'] == 'MEDIUM':
                st.markdown(f'<div class="alert-medium">⚠️ Medium Risk - Review Required</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low">✅ Low Risk - Transaction Approved</div>', 
                           unsafe_allow_html=True)
    
    # Model information
    with st.expander("📊 Model Information"):
        st.write("""
        **Fraud Detection System Details:**
        - **Models Used**: XGBoost + Isolation Forest Ensemble
        - **Training Data**: 284,807 credit card transactions
        - **Model Accuracy**: 98.31% ROC-AUC Score
        - **Fraud Detection Rate**: 85% Recall
        - **Processing Speed**: <100ms per transaction
        - **Real-time Capability**: 1000+ transactions/second
        """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
