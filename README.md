# 💳 Real-Time Credit Card Fraud Detection (Kubernetes Deployment)

## 🚀 Live Demo
👉 https://<your-streamlit-url>
*Note: The demo is available only while the local Kubernetes cluster is running.*

## ⚙️ How It Works
```text
Streamlit Community Cloud
        │
        ▼
      ngrok
        │
        ▼
Kubernetes Service
        │
        ▼
FastAPI Inference API
        │
        ▼
XGBoost Model
```

The model is pre-trained and deployed as a **Dockerized FastAPI service** running on **Kubernetes**. Streamlit sends transaction data to the API and displays real-time fraud predictions.

---

# Overview

Production-style credit card fraud detection system built to handle **extreme class imbalance (~0.17% fraud cases)**. The project demonstrates an end-to-end ML workflow from data preprocessing and model training to **containerized model serving with Kubernetes**.

---

# Problem Statement

Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent transactions are rare but costly.

**Dataset (Kaggle – Credit Card Fraud Detection)**

- **Rows:** 284,807
- **Features:** Time, V1–V28, Amount
- **Target:** Class (Fraud / Legit)

| Class | Count |
|------:|------:|
| Fraud | 473 |
| Legit | 283,253 |

Fraud transactions represent approximately **0.17%** of the dataset.

---

# Data Preprocessing

**Platform:** Databricks Community Edition

**Tools:** Spark + SQL

- Data validation and cleanup
- Feature selection
- Export processed dataset
- Prepare training data for XGBoost

---

# Model Training

**Platform:** AWS EC2 (CPU, Free Tier)

**Model:** XGBoost

Techniques:
- `scale_pos_weight`
- Stratified 5-fold cross-validation
- Threshold tuning
- Standard scaling

### Performance

| Metric | Value |
|--------|------:|
| Mean CV ROC-AUC | **0.9771** |
| ROC-AUC | **0.9973** |
| Precision | **0.8878** |
| Recall | **0.9704** |

---

# Kubernetes Model Serving

The inference service is deployed as a **Dockerized FastAPI application** on **Kubernetes**.

Pipeline:

```text
Docker Image
      │
      ▼
Kubernetes Deployment
      │
      ▼
Pod
      │
      ▼
Kubernetes Service
      │
      ▼
FastAPI REST API
```

The Streamlit application consumes the REST API for real-time inference.

---

# Streamlit UI

Features:

- Transaction input form
- Real-time fraud prediction
- Fraud probability
- Fraud / Legit label
- Prediction history

---

# Tech Stack

- Python
- XGBoost
- Databricks
- Spark
- SQL
- AWS EC2 (Training)
- Docker
- FastAPI
- Kubernetes
- Streamlit

---

# Key Takeaways

- End-to-end ML pipeline
- Production-style model serving
- Dockerized inference API
- Kubernetes deployment
- Interactive Streamlit frontend
- Designed for Machine Learning Engineer portfolio


-----------------------------------------------
Author: Iskandar Kholmanov
Role Target: Machine Learning Engineer / Applied Scientist / Senior Data Scientist

