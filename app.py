import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

from qa_config import (
    ModelQA,
    FraudModelQA,
    RAGQA,
    VectorStoreQA,
    ValidationResult,
    ValidationStatus
)

# =============================================================================
# Configuration & Theme
# =============================================================================

st.set_page_config(
    page_title="Smart-Insure AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Normalized CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main {
        background: #f8fafc;
    }
    
    /* Normalized Card Buttons */
    div.stButton > button {
        height: 80px;
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: #f8fafc;
    }

    /* Tab styling - Normalized */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f1f5f9;
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 8px 16px;
        color: #64748b;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #4f46e5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    /* Metrics */
    div.stMetric {
        background: #e0f2fe;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #bae6fd;
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
    }

    div.stMetric [data-testid="stMetricValue"] {
        color: #0369a1;
    }

    /* Forms and Containers */
    [data-testid="stForm"], .stForm {
        background: #ffffff;
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    div.stVerticalBlock > div > div.stBox {
        background: #f8fafc;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1rem;
    }

    /* Custom Card Variants */
    .customer-card div.stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .customer-card div.stButton > button:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
    }

    .company-card div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    }
    .company-card div.stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    }

    .log-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .header-text {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# State Management
# =============================================================================

if 'user_role' not in st.session_state:
    st.session_state.user_role = None

def select_role(role):
    st.session_state.user_role = role

# =============================================================================
# Storage & Models
# =============================================================================

CLAIMS_FILE = "claims_storage.json"

def save_claim(data, role, prediction=None):
    claim_entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "data": data,
        "prediction": prediction
    }
    claims = []
    if os.path.exists(CLAIMS_FILE):
        with open(CLAIMS_FILE, "r") as f:
            try: claims = json.load(f)
            except: claims = []
    claims.append(claim_entry)
    with open(CLAIMS_FILE, "w") as f:
        json.dump(claims, f, indent=4)
    return claim_entry["id"]

@st.cache_resource
def load_regression_model():
    model_path = "insurance_model.pkl"
    if os.path.exists(model_path): return joblib.load(model_path)
    return None

@st.cache_resource
def load_fraud_model_data():
    model_path = "fraud_model.pkl"
    if os.path.exists(model_path): return joblib.load(model_path)
    return None

@st.cache_resource
def load_vector_db():
    db_path = "faiss_index"
    if not os.path.exists(db_path): return None
    try:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
    except: return None

@st.cache_resource
def load_qa_chain(_db):
    if _db is None: return None
    try:
        llm = Ollama(model="phi3")
        return RetrievalQA.from_chain_type(llm=llm, retriever=_db.as_retriever())
    except: return None

def predict_fraud(model_data, input_data):
    model = model_data['model']
    feature_names = model_data['feature_names']
    df_input = pd.DataFrame([input_data])
    df_encoded = pd.get_dummies(df_input, columns=['accident_type'])
    for col in feature_names:
        if col not in df_encoded.columns: df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]
    prediction = model.predict(df_encoded)[0]
    probability = model.predict_proba(df_encoded)[0][1]
    return prediction, probability

# =============================================================================
# Dashboard / Landing Page
# =============================================================================

def show_landing_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;' class='header-text'>Smart-Insure AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;' class='subtitle-text'>Welcome. Please select your portal to continue.</p>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns([1.5, 1.5, 0.2, 1.5, 1.5])
    
    with col2:
        st.markdown('<div class="customer-card">', unsafe_allow_html=True)
        if st.button("Customer Portal", key="btn_cust"):
            select_role("Customer")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col4:
        st.markdown('<div class="company-card">', unsafe_allow_html=True)
        if st.button("Company Console", key="btn_comp"):
            select_role("Company")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Main Application Logic
# =============================================================================

if st.session_state.user_role is None:
    show_landing_page()
else:
    # Sidebar back button
    st.sidebar.markdown("<h3 class='header-text'>Workspace</h3>", unsafe_allow_html=True)
    if st.sidebar.button("Switch Account", use_container_width=True):
        st.session_state.user_role = None
        st.rerun()
    
    # Load Resources
    reg_model = load_regression_model()
    fraud_data = load_fraud_model_data()
    vector_db = load_vector_db()
    qa_chain = load_qa_chain(vector_db)

    if st.session_state.user_role == "Customer":
        st.title("Customer Portal")
        
        tab1, tab2, tab3 = st.tabs(["Claims", "Estimates", "Support"])
        
        with tab1:
            st.subheader("Submit Claim")
            with st.form("claim_form"):
                col1, col2 = st.columns(2)
                with col1:
                    claim_amount = st.number_input("Claim Amount ($)", 0, 10000000, 5000)
                    vehicle_age = st.number_input("Vehicle Age", 0, 30, 5)
                    accident_type = st.selectbox("Accident Type", ["Rear-end", "Side-swipe", "Front-end", "Parked Car"])
                    police_report = st.selectbox("Police Report Filed", ["No", "Yes"])
                with col2:
                    witness_present = st.selectbox("Witness Present", ["No", "Yes"])
                    previous_claims = st.number_input("Previous Claims", 0, 50, 0)
                    premium = st.number_input("Annual Premium ($)", 0, 100000, 1200)
                    insured_value = st.number_input("Insured Value ($)", 0, 20000000, 25000)
                
                submitted = st.form_submit_button("Submit Claim")
                if submitted:
                    input_features = {
                        "claim_amount": claim_amount, "vehicle_age": vehicle_age,
                        "accident_type": accident_type, "police_report": 1 if police_report == "Yes" else 0,
                        "witness_present": 1 if witness_present == "Yes" else 0,
                        "previous_claims": previous_claims, "premium": premium, "insured_value": insured_value
                    }
                    validation_results = FraudModelQA.validate_features(input_features)
                    failures = [r for r in validation_results if r.status == ValidationStatus.FAIL]
                    
                    if failures:
                        for f in failures: st.error(f"Error: {f.message}")
                    else:
                        claim_id = save_claim(input_features, "Customer")
                        st.success(f"Claim filed. ID: {claim_id}")
        
        with tab2:
            st.subheader("Coverage Estimation")
            if reg_model:
                col1, col2 = st.columns(2)
                with col1:
                    c_sex = st.selectbox("Sex", ["Male", "Female"])
                    c_type = st.number_input("Insurance Type", 1, 10, 1)
                    c_val = st.number_input("Vehicle Value", 0, 10000000, 25000)
                    c_prem = st.number_input("Premium", 0, 100000, 1500)
                with col2:
                    c_year = st.number_input("Year", 1990, 2025, 2020)
                    c_seats = st.number_input("Seats", 1, 50, 5)
                    c_ccm = st.number_input("CCM", 0, 20000, 1600)
                    c_make = st.number_input("Make ID", 0, 1000, 1)
                
                if st.button("Calculate", type="primary"):
                    sex_bit = 1 if c_sex == "Male" else 0
                    vector = [[sex_bit, 1, 1, 2024, c_type, c_val, c_prem, 12345, c_year, c_seats, 500, 1, c_ccm, c_make, 1]]
                    prediction = reg_model.predict(vector)[0]
                    st.metric("Estimated Amount", f"${prediction:,.2f}")
            else: st.error("Model unavailable.")

        with tab3:
            st.subheader("Policy Support")
            query = st.text_input("How can we help?")
            if query:
                if qa_chain:
                    with st.spinner("Searching..."):
                        st.write(qa_chain.run(query))
                else: st.warning("Engine offline.")

    else: # Company View
        st.title("Underwriter Console")
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("Risk Analysis")
            with st.container():
                c_amount = st.number_input("Amount", 0, 10000000, 5000, key="ca1")
                v_age = st.number_input("Age", 0, 30, 5, key="ca2")
                acc_type = st.selectbox("Category", ["Rear-end", "Side-swipe", "Front-end", "Parked Car"], key="ca3")
                pol_rep = st.selectbox("Police Report", ["No", "Yes"], key="ca4")
                wit_pres = st.selectbox("Witness", ["No", "Yes"], key="ca5")
                prev_claims = st.number_input("History", 0, 50, 0, key="ca6")
                prem = st.number_input("Premium", 0, 100000, 1200, key="ca7")
                ins_val = st.number_input("Value", 0, 20000000, 25000, key="ca8")
                
                if st.button("Analyze Risk", type="primary"):
                    input_features = {
                        "claim_amount": c_amount, "vehicle_age": v_age, "accident_type": acc_type,
                        "police_report": 1 if pol_rep == "Yes" else 0, "witness_present": 1 if wit_pres == "Yes" else 0,
                        "previous_claims": prev_claims, "premium": prem, "insured_value": ins_val
                    }
                    if fraud_data:
                        pred, prob = predict_fraud(fraud_data, input_features)
                        save_claim(input_features, "Company", {"result": int(pred), "confidence": float(prob)})
                        
                        st.divider()
                        if pred == 1:
                            st.error(f"High Risk Detected ({prob:.2%})")
                        else:
                            st.success(f"Claim Verified ({1-prob:.2%})")
                    else: st.error("Module missing.")

        with col2:
            st.subheader("Submission Log")
            st.markdown('<div class="log-container">', unsafe_allow_html=True)
            if os.path.exists(CLAIMS_FILE):
                with open(CLAIMS_FILE, "r") as f:
                    claims = json.load(f)
                    df = pd.DataFrame([
                        {"Ref": c["id"][:8], "Role": c["role"], "Amount": f"${c['data']['claim_amount']:,}"} 
                        for c in claims[::-1]
                    ])
                    st.table(df.head(10))
            st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Smart-Insure AI v3.3")
