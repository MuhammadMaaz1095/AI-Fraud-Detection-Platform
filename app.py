from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import random

app = FastAPI(title="SentinelAI Real-World Fraud API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    sim_pool = pd.read_csv('simulation_pool.csv')
except FileNotFoundError:
    raise RuntimeError("Required model files or simulation_pool.csv missing. Run model.py first!")

class TransactionPayload(BaseModel):
    tx_index: int


historical_ledger = []

@app.get("/api/v1/pool")
def get_simulation_pool():
   
    records = []
    for idx, row in sim_pool.iterrows():
        records.append({
            "index": int(idx),
            "amount": float(row['Amount']),
            "time": float(row['Time']),
            "actual_class": int(row['Class'])
        })
    return records

@app.post("/api/v1/score")
def score_transaction(payload: TransactionPayload):
    if payload.tx_index not in sim_pool.index:
        raise HTTPException(status=404, detail="Transaction index outside simulation boundary pool.")
    
    
    raw_row = sim_pool.loc[[payload.tx_index]].drop(columns=['Class'])
    
    
    scaled_array = scaler.transform(raw_row)
    scaled_df = pd.DataFrame(scaled_array, columns=raw_row.columns)
    
   
    prob = float(model.predict_proba(scaled_df)[0][1])
    risk_score = int(prob * 100)
    
   
    shap_vals = explainer.shap_values(scaled_df)[0]
    
   
    feature_impacts = {col: float(val) for col, val in zip(raw_row.columns, shap_vals)}
    sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_5_explanations = dict(sorted_impacts)
    
  
    primary_driver = max(top_5_explanations, key=top_5_explanations.get)
    status = "FLAGGED" if risk_score >= 50 else "APPROVED"
    

    if status == "FLAGGED":
        reason = f"Anomalous variance detected across component {primary_driver}. High feature score correlation indicating fraud footprint match."
    else:
        reason = "All features behave consistently within regular parameters."

    result = {
        "id": f"TX-KGL{random.randint(10000, 99999)}",
        "amount": float(raw_row['Amount'].values[0]),
        "time": float(raw_row['Time'].values[0]),
        "risk_score": risk_score,
        "status": status,
        "reason": reason,
        "shap_explanations": top_5_explanations
    }
    
    historical_ledger.insert(0, result)
    return result

@app.get("/api/v1/dashboard")
def get_dashboard_metrics():
    total = len(historical_ledger)
    flagged = [t for t in historical_ledger if t["status"] == "FLAGGED"]
    return {
        "metrics": {
            "total_processed": total,
            "flagged_count": len(flagged),
            "fraud_rate": round((len(flagged) / total) * 100, 1) if total > 0 else 0,
            "total_prevented_loss": sum([t["amount"] for t in flagged])
        },
        "recent_transactions": historical_ledger[:10]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)