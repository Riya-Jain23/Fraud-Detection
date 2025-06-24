import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    def __init__(self):
        self.xgb_model = None
        self.isolation_forest = None
        self.feature_columns = None
        self.is_trained = False
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare the credit card dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        print(f"Data loaded: {X.shape[0]} transactions, {X.shape[1]} features")
        print(f"Fraud cases: {sum(y)} ({(sum(y)/len(y)*100):.3f}%)")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """Train both XGBoost and Isolation Forest models"""
        print("\n" + "="*50)
        print("TRAINING FRAUD DETECTION MODELS")
        print("="*50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} transactions")
        print(f"Test set: {len(X_test)} transactions")
        
        # 1. Train Isolation Forest (Unsupervised)
        print("\n1. Training Isolation Forest...")
        self.isolation_forest = IsolationForest(
            contamination=0.002,  # Expected fraud rate
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_train)
        print("✅ Isolation Forest trained!")
        
        # 2. Handle imbalanced data with SMOTE
        print("\n2. Balancing dataset with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {len(X_train_balanced)} balanced samples")
        
        # 3. Train XGBoost (Supervised)
        print("\n3. Training XGBoost...")
        self.xgb_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.xgb_model.fit(X_train_balanced, y_train_balanced)
        print("✅ XGBoost trained!")
        
        # 4. Evaluate models
        self._evaluate_models(X_test, y_test)
        
        self.is_trained = True
        return X_test, y_test
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate both models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # XGBoost predictions
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Isolation Forest predictions
        iso_pred = self.isolation_forest.predict(X_test)
        iso_pred_binary = (iso_pred == -1).astype(int)  # -1 = anomaly = fraud
        
        print("XGBoost Results:")
        print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_prob):.4f}")
        print(classification_report(y_test, xgb_pred))
        
        print("\nIsolation Forest Results:")
        print(classification_report(y_test, iso_pred_binary))
        
        return {
            'xgb_auc': roc_auc_score(y_test, xgb_prob),
            'xgb_predictions': xgb_pred,
            'iso_predictions': iso_pred_binary
        }
    
    def predict_fraud_probability(self, X):
        """Get fraud probability for new transactions"""
        if not self.is_trained:
            raise Exception("Models not trained yet!")
        
        # XGBoost probability
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        
        # Isolation Forest anomaly score
        iso_score = self.isolation_forest.decision_function(X)
        iso_prob = 1 / (1 + np.exp(iso_score))  # Convert to probability-like score
        
        # Combined ensemble score
        combined_prob = (xgb_prob * 0.7) + (iso_prob * 0.3)
        
        return combined_prob
    
    def predict_single_transaction(self, transaction_data):
        """Predict fraud for a single transaction"""
        # Convert to DataFrame if it's a dict
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = pd.DataFrame([transaction_data], columns=self.feature_columns)
        
        fraud_prob = self.predict_fraud_probability(df)[0]
        
        # Risk categorization
        if fraud_prob < 0.3:
            risk_level = "LOW"
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        return {
            'fraud_probability': fraud_prob,
            'risk_level': risk_level,
            'is_fraud': fraud_prob > 0.5
        }
    
    def save_models(self, filepath='fraud_detection_models.pkl'):
        """Save trained models"""
        if not self.is_trained:
            raise Exception("No trained models to save!")
        
        model_data = {
            'xgb_model': self.xgb_model,
            'isolation_forest': self.isolation_forest,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath='fraud_detection_models.pkl'):
        """Load pre-trained models"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.xgb_model = model_data['xgb_model']
        self.isolation_forest = model_data['isolation_forest']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        print(f"✅ Models loaded from {filepath}")

def main():
    """Train the fraud detection system"""
    # Initialize detector
    detector = FraudDetector()
    
    # Load data (adjust path as needed)
    X, y = detector.load_and_prepare_data('../data/creditcard.csv')
    
    # Train models
    X_test, y_test = detector.train_models(X, y)
    
    # Save models
    detector.save_models()
    
    # Test single transaction
    print("\n" + "="*50)
    print("TESTING SINGLE TRANSACTION")
    print("="*50)
    
    # Get a sample transaction
    sample_transaction = X_test.iloc[0].to_dict()
    result = detector.predict_single_transaction(sample_transaction)
    
    print(f"Transaction Analysis:")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Predicted as Fraud: {result['is_fraud']}")
    
    print("\n🎉 Fraud Detection System Ready!")

if __name__ == "__main__":
    main()