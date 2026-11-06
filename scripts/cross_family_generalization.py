import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from xgboost import XGBClassifier

print("="*60)
print("Cross-Family Generalization Experiment")
print("="*60)

print("\n[1/5] Loading data...")
X_raw = pd.read_csv('data/processed/X_features.csv')
y = pd.read_csv('data/processed/y_labels.csv').squeeze()
labels_df = pd.read_csv('data/processed/quantum_labels.csv')

# Clean non-numeric
non_numeric = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
X_raw = X_raw.drop(columns=non_numeric)
X_raw = X_raw.fillna(0)

# Fix column names
X_raw.columns = [str(c).replace('[','_').replace(']','_')
                       .replace('<','_').replace('>','_')
                       .replace(' ','_') for c in X_raw.columns]

# Drop direct labels
direct_label_cols = [c for c in X_raw.columns if c.startswith('Q')]
X_bytecode = X_raw.drop(columns=direct_label_cols)

print("\n[2/5] Creating OpenZeppelin and Custom ECDSA subsets...")
# Get indices for OZ and Custom
idx_oz = labels_df.index[labels_df['Q2_oz_ecdsa_detected'] == 1].tolist()
idx_custom = labels_df.index[(labels_df['Q1_ecrecover_detected'] == 1) & (labels_df['Q2_oz_ecdsa_detected'] == 0)].tolist()

# Get indices for safe contracts
idx_safe = labels_df.index[labels_df['quantum_vulnerable'] == 0].tolist()

# Sample safe contracts to balance
np.random.seed(42)
idx_safe_oz = np.random.choice(idx_safe, size=len(idx_oz), replace=False).tolist()
# Remove used safe contracts for the custom set
remaining_safe = list(set(idx_safe) - set(idx_safe_oz))
idx_safe_custom = np.random.choice(remaining_safe, size=len(idx_custom), replace=False).tolist()

print(f"  OpenZeppelin positives: {len(idx_oz)}, safe: {len(idx_safe_oz)}")
print(f"  Custom ECDSA positives: {len(idx_custom)}, safe: {len(idx_safe_custom)}")

# Datasets
train_oz_idx = idx_oz + idx_safe_oz
X_oz = X_bytecode.iloc[train_oz_idx]
y_oz = y.iloc[train_oz_idx]

train_custom_idx = idx_custom + idx_safe_custom
X_custom = X_bytecode.iloc[train_custom_idx]
y_custom = y.iloc[train_custom_idx]

# Scale
scaler_oz = StandardScaler()
X_oz_sc = scaler_oz.fit_transform(X_oz)
X_custom_sc1 = scaler_oz.transform(X_custom)

scaler_custom = StandardScaler()
X_custom_sc2 = scaler_custom.fit_transform(X_custom)
X_oz_sc2 = scaler_custom.transform(X_oz)

print("\n[3/5] Experiment A: Train on OpenZeppelin, Test on Custom ECDSA")
model_a = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                        eval_metric='logloss', random_state=42, verbosity=0)
model_a.fit(X_oz_sc, y_oz)
y_pred_a = model_a.predict(X_custom_sc1)
y_prob_a = model_a.predict_proba(X_custom_sc1)[:,1]

print(f"  Accuracy : {accuracy_score(y_custom, y_pred_a):.4f}")
print(f"  F1 Score : {f1_score(y_custom, y_pred_a):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_custom, y_prob_a):.4f}")

print("\n[4/5] Experiment B: Train on Custom ECDSA, Test on OpenZeppelin")
model_b = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                        eval_metric='logloss', random_state=42, verbosity=0)
model_b.fit(X_custom_sc2, y_custom)
y_pred_b = model_b.predict(X_oz_sc2)
y_prob_b = model_b.predict_proba(X_oz_sc2)[:,1]

print(f"  Accuracy : {accuracy_score(y_oz, y_pred_b):.4f}")
print(f"  F1 Score : {f1_score(y_oz, y_pred_b):.4f}")
print(f"  AUC-ROC  : {roc_auc_score(y_oz, y_prob_b):.4f}")

print("\n[5/5] Conclusion")
auc_a = roc_auc_score(y_custom, y_prob_a)
auc_b = roc_auc_score(y_oz, y_prob_b)

if auc_a > 0.8 and auc_b > 0.8:
    print("✅ SUCCESS: The model successfully generalizes across different implementations.")
    print("This proves the model learns structural representations rather than memorizing labels!")
else:
    print("❌ FAILURE: Generalization performance dropped.")
    print("The model might be overfitting to specific implementation details.")

print("\nDone.")
