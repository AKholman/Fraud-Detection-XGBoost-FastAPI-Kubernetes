#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import json, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


# In[ ]:


# Load data
df = pd.read_csv("processed_data.csv")
X = df.drop("Class", axis=1)
y = df["Class"]


# In[ ]:


# Save feature list
features = list(X.columns)
json.dump(features, open("feature_list.json", "w"))


# In[ ]:


# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()


# In[ ]:


model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=1   # CPU only
)

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_scores = []

for train_idx, val_idx in skf.split(X_scaled, y):
    model.fit(X_scaled[train_idx], y.iloc[train_idx])
    preds = model.predict_proba(X_scaled[val_idx])[:, 1]
    roc_scores.append(roc_auc_score(y.iloc[val_idx], preds))

print("Mean ROC-AUC:", sum(roc_scores) / len(roc_scores))


# In[ ]:


# Threshold tuning
probs = model.predict_proba(X_scaled)[:, 1]
threshold = 0.3
preds = (probs >= threshold).astype(int)

print("Precision:", precision_score(y, preds))
print("Recall:", recall_score(y, preds))
print("ROC-AUC:", roc_auc_score(y, probs))

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

