import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("⏳ Loading real-world Kaggle credit card dataset...")
try:
    df = pd.read_csv('creditcard.csv')
except FileNotFoundError:
    raise FileNotFoundError("🔴 'creditcard.csv' not found! Download it from Kaggle and place it in this folder.")

X = df.drop(columns=['Class'])
y = df['Class']

print(f"📊 Dataset Loaded: {X.shape[0]} rows | Total Fraud Cases: {y.sum()} ({round((y.sum()/len(y))*100, 3)}%)")

# Stratified Split to preserve the tiny minority fraud class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

imbalance_ratio = (len(y_train) - sum(y_train)) / sum(y_train)

print("🚀 Training robust XGBoost Engine on real financial patterns...")
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=imbalance_ratio, 
    eval_metric='aucpr',            
    random_state=42
)
model.fit(X_train_scaled_df, y_train)

print("🧠 Generating SHAP Explainer tree...")
explainer = shap.TreeExplainer(model)

print("💾 Saving structural runtime artifacts...")
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

simulation_pool = pd.concat([
    X_test[y_test == 0].sample(20, random_state=42),
    X_test[y_test == 1].sample(10, random_state=42)
], axis=0)
simulation_pool['Class'] = y_test.loc[simulation_pool.index]
simulation_pool.to_csv('simulation_pool.csv', index=False)

print("✅ Real-world Model artifacts & Simulation Pool successfully initialized!")