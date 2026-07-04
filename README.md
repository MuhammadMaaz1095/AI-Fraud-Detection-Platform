# SentinelAI: End-to-End Real-World Financial Fraud Detection Platform

SentinelAI is an enterprise-grade financial intelligence application designed for risk analysts and security operations centers. The platform leverages a high-performance **XGBoost machine learning engine** trained on authentic, highly imbalanced real-world financial transaction distributions. It integrates **FastAPI** for low-latency scoring pipelines and features an interactive web dashboard complete with localized **SHAP (SHapley Additive exPlanations)** explainability metrics to bridge the gap between black-box AI and human decision-making.

---

## 🚀 Core Architectural Features

* **Real-World Engine Core**: Built utilizing the gold-standard ULB Kaggle Credit Card Fraud Detection dataset (284,807 production records) featuring extreme class imbalance ($0.17\%$ fraud prevalence).
* **Imbalance-Aware AI Architecture**: Implements advanced data scaling alongside stratified dataset partitioning, deploying custom training weights (`scale_pos_weight`) inside the XGBoost framework to optimize the **Precision-Recall Curve (AUCPR)**.
* **Explainable AI (XAI)**: Connects a `TreeExplainer` matrix directly into the transactional scoring API, passing feature-level contribution metrics (SHAP vectors) to decode why an alert was generated.
* **Production Interface Stack**: A sleek, dark-theme cybersecurity dashboard styled with TailwindCSS and Lucide vectors, featuring an interactive live validation pool, a simulated security credential portal, and dynamic SHAP feature graphs.

---

## 🛠️ Tech Stack & Dependencies

* **Language Platform**: Python 3.10+
* **Machine Learning**: `xgboost`, `shap`, `scikit-learn`, `pandas`, `numpy`
* **Backend Framework**: `fastapi`, `uvicorn`, `pydantic`
* **Frontend Web Deck**: HTML5, TailwindCSS (CDN), Lucide-Icons, Native JavaScript ES6

---

## 📁 Repository Directory Structure

```text
capston_project/
├── .venv/                  # Python isolated environment (Ignored in Git)
├── creditcard.csv          # Real Kaggle dataset (Ignored in Git)
├── fraud_model.pkl         # Serialized XGBoost AI weights (Ignored in Git)
├── scaler.pkl              # Serialized StandardScaler boundaries (Ignored in Git)
├── explainer.pkl           # Serialized SHAP explainer configurations (Ignored in Git)
├── simulation_pool.csv     # Curated evaluation pool extracted from test set
├── model.py                # ML Training & Data Preprocessing Pipeline
├── app.py                  # FastAPI Backend Stream Server Routing Script
├── login.html              # Secure Operator Authorization Entrance Portal
└── index.html              # Central Command Analytics & Explainer Dashboard
