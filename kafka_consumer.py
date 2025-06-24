from kafka import KafkaConsumer
import sys
import os

# Add parent directory to path to import the fraud detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fraud_detector import FraudDetector

# Load the fraud detection model
detector = FraudDetector()
detector.load_models('./models/fraud_detection_models.pkl')

# Kafka consumer configuration
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-detection-group'
)

# Process transactions from Kafka
for message in consumer:
    transaction = message.value
    result = detector.predict_single_transaction(transaction)
    print(f"Transaction: {transaction}, Result: {result}")