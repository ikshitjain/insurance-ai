# Smart-Insure AI 🤖🛡️

Smart-Insure AI ek advanced end-to-end Insurance Intelligence System hai jo Machine Learning aur RAG (Retrieval-Augmented Generation) ka use karke claims fraud detection, coverage cost estimation aur automated policy support pradan karta hai. Is project me ek robust FastAPI backend aur ek interactive Streamlit web application shamil hai.

**Live App Link:** [Smart-Insure AI Streamlit App](https://insurance-ai-ftetfqw6kfruveatuq9yjfbyik.streamlit.app/)

---

## 📌 Features & Portals

Project ko do mukhya workflows me banta gaya hai, jise user-role ke anusar access kiya ja sakta hai:

### 1. Customer Portal 👤
* **Submit Claim:** Users apne vehicle claims submit kar sakte hain. Submit karne se pehle validation engine data ki quality check karta hai.
* **Coverage Estimation:** Machine Learning model ke dwara features (jaise vehicle value, premium, seating capacity, ccm, etc.) ke aadhar par estimated coverage amount calculate karta hai.
* **Policy Support (RAG):** Groq (Llama-3.1) aur FAISS vector database ka use karke users policy documents ke sawal-jawab (Q&A) real-time me kar sakte hain.

### 2. Underwriter Console (Company View) 🏢
* **Risk Analysis & Fraud Detection:** Underwriters claims data ko analyze kar sakte hain. System predictive model ka use karke fraud probability (%) aur risk status (High Risk vs Verified) output deta hai.
* **Submission Log:** Sabhi portal se submit huye claims ka ek centralized digital log track hota hai aur tabular format me dikhai deta hai.

---

## 📂 Project Architecture & File Structure

```text
├── faiss_index/            # RAG ke liye FAISS Vector store database binary files
│   ├── index.faiss
│   └── index.pkl
├── data/
│   └── PCIPolicyWording.pdf# Base policy document jise vector store me embed kiya gaya hai
├── qa_config.py            # Quality Assurance thresholds aur strict evaluation validators
├── api.py                  # Production-ready FastAPI endpoints implementation
├── app.py                  # Responsive UI ke sath Streamlit frontend multi-tab UI
├── run_all.py              # Backend aur Frontend ko ek sath start karne ki utility script
├── requirements.txt        # Python libraries dependencies specification
├── insurance_model.pkl     # Coverage estimation ke liye trained Regressor model
├── fraud_model.pkl         # Fraud classification ke liye trained Scikit-Learn model data
└── claims_storage.json     # Submissions aur predictions data ko persist karne ke liye local JSON storage
