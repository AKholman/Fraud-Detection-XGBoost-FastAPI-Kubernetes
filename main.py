from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json

app = FastAPI(title="Fraud Detection API")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

with open("feature_list.json") as f:
    features = json.load(f)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.model_dump()])[features]
    X = scaler.transform(df)

    prob = float(model.predict_proba(X)[0][1])
    label = int(prob >= 0.3)

    return {
        "fraud_probability": round(prob, 4),
        "fraud_label": label
    }