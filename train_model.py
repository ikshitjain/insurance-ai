import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_fraud_model():
    # 1. Load dataset
    if not os.path.exists("fraud_dataset.csv"):
        print("Error: fraud_dataset.csv not found.")
        return

    df = pd.read_csv("fraud_dataset.csv")
    print("Dataset loaded successfully.")

    # 2. Handle missing values (none expected in synthetic, but good practice)
    df = df.fillna(df.median(numeric_only=True))

    # 3. Convert categorical variables to numeric (One-Hot Encoding)
    # accident_type is the only categorical column
    df_encoded = pd.get_dummies(df, columns=['accident_type'])
    
    # 4. Split into train/test
    X = df_encoded.drop('fraud_reported', axis=1)
    y = df_encoded['fraud_reported']
    
    # Save feature names for consistency during prediction
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train RandomForestClassifier
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save model and metadata
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'categorical_columns': ['accident_type'] # Original cat cols
    }
    
    joblib.dump(model_data, "fraud_model.pkl")
    print(f"\nModel saved as fraud_model.pkl")

if __name__ == "__main__":
    train_fraud_model()
