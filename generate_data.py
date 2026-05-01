import pandas as pd
import numpy as np
import os

def generate_fraud_dataset(n_samples=1200):
    np.random.seed(42)
    
    # Features
    claim_amount = np.random.randint(500, 2000000, n_samples)
    vehicle_age = np.random.randint(0, 25, n_samples)
    accident_type = np.random.choice(['Rear-end', 'Side-swipe', 'Front-end', 'Parked Car'], n_samples)
    police_report = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    witness_present = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    previous_claims = np.random.randint(0, 10, n_samples)
    premium = np.random.randint(500, 5000, n_samples)
    insured_value = claim_amount * np.random.uniform(1.1, 2.5, n_samples)
    
    # Logic for Fraud (Simplified)
    # Higher chance of fraud if: high claim, no police report, high previous claims, no witness
    fraud_score = (
        0.3 * (claim_amount / 2000000) +
        0.3 * (1 - police_report) +
        0.2 * (previous_claims / 10) +
        0.2 * (1 - witness_present)
    )
    
    fraud_reported = (fraud_score > 0.6).astype(int)
    
    df = pd.DataFrame({
        'claim_amount': claim_amount,
        'vehicle_age': vehicle_age,
        'accident_type': accident_type,
        'police_report': police_report,
        'witness_present': witness_present,
        'previous_claims': previous_claims,
        'premium': premium,
        'insured_value': insured_value,
        'fraud_reported': fraud_reported
    })
    
    output_path = "fraud_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path} ({n_samples} rows)")

if __name__ == "__main__":
    generate_fraud_dataset()
