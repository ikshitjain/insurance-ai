import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    age = np.random.randint(18, 75, n_samples)
    policy_annual_premium = np.random.randint(500, 3000, n_samples)
    insured_sex = np.random.randint(0, 2, n_samples)
    incident_severity = np.random.randint(0, 3, n_samples)  # 0: Minor, 1: Major, 2: Total Loss
    property_damage = np.random.randint(0, 2, n_samples)
    police_report_available = np.random.randint(0, 2, n_samples)
    
    total_claim_amount = np.random.randint(500, 50000, n_samples)
    injury_claim = total_claim_amount * np.random.uniform(0.1, 0.3, n_samples)
    property_claim = total_claim_amount * np.random.uniform(0.1, 0.3, n_samples)
    vehicle_claim = total_claim_amount - injury_claim - property_claim
    
    # Generate Target (Fraud)
    # Higher chance of fraud if high claim amount, no police report, and total loss
    fraud_prob = (
        0.1 * (total_claim_amount / 50000) +
        0.3 * (2 - incident_severity) / 2 + # Lower severity but high claim
        0.3 * (1 - police_report_available) +
        0.1 * (property_damage)
    )
    # Normalize prob
    fraud_prob = (fraud_prob - fraud_prob.min()) / (fraud_prob.max() - fraud_prob.min())
    fraud = (fraud_prob > 0.7).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'policy_annual_premium': policy_annual_premium,
        'insured_sex': insured_sex,
        'incident_severity': incident_severity,
        'property_damage': property_damage,
        'police_report_available': police_report_available,
        'total_claim_amount': total_claim_amount,
        'injury_claim': injury_claim,
        'property_claim': property_claim,
        'vehicle_claim': vehicle_claim,
        'fraud': fraud
    })
    
    return data

def train_and_save_model():
    print("Generating synthetic data...")
    data = generate_synthetic_data(2000)
    
    X = data.drop('fraud', axis=1)
    y = data['fraud']
    
    print("Training Fraud Detection Model (RandomForest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    model_path = "fraud_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names for reference
    feature_names = X.columns.tolist()
    print(f"Features: {feature_names}")
    return model

if __name__ == "__main__":
    train_and_save_model()
