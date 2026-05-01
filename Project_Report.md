# Project Report: Smart-Insure AI
**An Integrated ML & RAG-based Vehicle Insurance Analysis System**

---

## 1. Executive Summary
Smart-Insure is a dual-intelligence AI system designed for the modern insurance industry. It combines **Machine Learning (ML)** for predictive analytics and **Generative AI (RAG)** for document intelligence. The system's unique strength lies in its integrated **Quality Assurance (QA) Framework**, ensuring reliability and data integrity.

## 2. Problem Statement
Manual insurance claim processing and policy interpretation are time-consuming and prone to errors. Customers often struggle to understand complex legal policy wordings, and insurance agents need quick, data-driven tools to estimate potential claim liabilities.

## 3. System Architecture
The project is built on three main pillars:
1. **Predictive Analytics Engine:** A Scikit-learn based regression model that estimates claim amounts based on 15 vehicle and policy parameters.
2. **Knowledge Retrieval Engine (RAG):** A LangChain-powered pipeline that uses FAISS and local LLMs (Phi-3/TinyDolphin) to answer questions directly from insurance PDF documents.
3. **QA & Validation Layer:** A dedicated configuration module (`qa_config.py`) that monitors system health and validates all user inputs.

## 4. Technical Stack
| Component | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **AI Framework** | LangChain |
| **Machine Learning** | Scikit-learn, Joblib, NumPy |
| **Vector Database** | FAISS |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) |
| **Local LLM Hub** | Ollama |
| **PDF Processing** | PyPDF |

## 5. Key Features
* **Automated Claim Prediction:** Predicts payout amounts using historical vehicle data.
* **Semantic Policy Search:** Answers natural language queries from `PCIPolicyWording.pdf`.
* **System Health Dashboard:** Real-time monitoring of Model, Database, and LLM status.
* **Input Sanity Checks:** Prevents errors by validating feature ranges (e.g., ensuring 'Production Year' is realistic).
* **Privacy-First AI:** Runs entirely on local hardware using Ollama, ensuring no sensitive insurance data is sent to the cloud.

## 6. Implementation Highlights
* **Efficient Indexing:** Large PDF documents are split into optimized chunks and indexed into a FAISS vector store for sub-second retrieval.
* **Robust Error Handling:** The system includes cached health checks to minimize performance overhead while ensuring maximum uptime.
* **User-Centric Design:** A clean Streamlit interface with clear indicators for prediction results and source document citations.

## 7. Future Roadmap
* Integration of Computer Vision for automated damage assessment from vehicle photos.
* Expansion to multi-policy support (Life, Health, and Property insurance).
* Real-time API integration with national vehicle databases.

## 8. Conclusion
Smart-Insure demonstrates a production-grade approach to AI. By combining mathematical prediction with linguistic intelligence and a rigorous QA layer, it provides a trustworthy tool for both insurance providers and policyholders.

---
**Prepared by:** [Your Name/Team Name]
**Date:** April 2026
