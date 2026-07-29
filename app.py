# app.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# Kubernetes Service URL
API_URL = os.getenv("API_URL", "https://cross-able-willing.ngrok-free.dev -> http://localhost:8000/predict")

st.set_page_config(
    page_title="Fraud Detection",
    layout="centered"
)

st.title("💳 Real-Time Fraud Detection")

st.subheader("Transaction Input")

features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

input_data = {}
cols = st.columns(4)

for i, feature in enumerate(features):
    with cols[i % 4]:
        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )

if "history" not in st.session_state:
    st.session_state.history = []


if st.button("Predict Fraud"):

    response = requests.post(
        API_URL,
        json=input_data
    )

    result = response.json()

    probability = result["fraud_probability"]
    label = result["fraud_label"]

    st.metric(
        "Fraud Probability",
        f"{probability:.4f}"
    )

    if label == 1:
        st.error("🚨 FRAUD")
    else:
        st.success("✅ LEGIT")

    st.session_state.history.append(probability)


# Prediction history chart
if st.session_state.history:

    history_df = pd.DataFrame({
        "Prediction": range(1, len(st.session_state.history)+1),
        "Fraud Probability": st.session_state.history
    })

    st.subheader("Prediction History")

    fig = px.line(
        history_df,
        x="Prediction",
        y="Fraud Probability"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )