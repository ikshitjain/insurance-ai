from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

app = FastAPI(title="Smart-Insure AI API")

# =============================================================================
# Models & Resources
# =============================================================================

# Cache for resources
resources = {
    "reg_model": None,
    "fraud_model_data": None,
    "vector_db": None,
    "qa_chain": None
}

def load_resources():
    # Load Regression Model
    reg_path = "insurance_model.pkl"
    if os.path.exists(reg_path):
        resources["reg_model"] = joblib.load(reg_path)
    
    # Load Fraud Model
    fraud_path = "fraud_model.pkl"
    if os.path.exists(fraud_path):
        resources["fraud_model_data"] = joblib.load(fraud_path)
    
    # Load Vector DB & QA Chain
    db_path = "faiss_index"
    if os.path.exists(db_path):
        try:
            embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            resources["vector_db"] = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
            
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name="llama3-8b-8192",
                    temperature=0
                )
                resources["qa_chain"] = RetrievalQA.from_chain_type(llm=llm, retriever=resources["vector_db"].as_retriever())
            else:
                print("GROQ_API_KEY not found in environment variables.")
        except Exception as e:
            print(f"Error loading RAG resources: {e}")

# Load on startup
@app.on_event("startup")
async def startup_event():
    load_resources()

# =============================================================================
# Schemas
# =============================================================================

class FraudInput(BaseModel):
    claim_amount: float
    vehicle_age: int
    accident_type: str
    police_report: int
    witness_present: int
    previous_claims: int
    premium: float
    insured_value: float

class CoverageInput(BaseModel):
    sex: str
    insr_type: int
    vehicle_value: float
    premium: float
    year: int
    seats: int
    ccm: float
    make_id: int

class QuestionInput(BaseModel):
    question: str

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def read_root():
    return {"status": "Smart-Insure AI API is running", "timestamp": datetime.now().isoformat()}

@app.post("/predict/fraud")
def predict_fraud_endpoint(data: FraudInput):
    if not resources["fraud_model_data"]:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")
    
    model_data = resources["fraud_model_data"]
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    input_dict = data.dict()
    df_input = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df_input, columns=['accident_type'])
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    df_encoded = df_encoded[feature_names]
    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "fraud_probability": float(probability),
        "status": "success"
    }

@app.post("/predict/coverage")
def predict_coverage_endpoint(data: CoverageInput):
    if not resources["reg_model"]:
        raise HTTPException(status_code=503, detail="Coverage model not loaded")
    
    sex_bit = 1 if data.sex.lower() == "male" else 0
    # Following the vector structure in app.py
    # [sex_bit, 1, 1, 2024, c_type, c_val, c_prem, 12345, c_year, c_seats, 500, 1, c_ccm, c_make, 1]
    vector = [[sex_bit, 1, 1, 2024, data.insr_type, data.vehicle_value, data.premium, 12345, data.year, data.seats, 500, 1, data.ccm, data.make_id, 1]]
    
    prediction = resources["reg_model"].predict(vector)[0]
    
    return {
        "estimated_amount": float(prediction),
        "status": "success"
    }

@app.post("/ask")
def ask_question(data: QuestionInput):
    if not resources["qa_chain"]:
        raise HTTPException(status_code=503, detail="QA system not available")
    
    try:
        answer = resources["qa_chain"].run(data.question)
        return {
            "answer": answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
