import pandas as pd
import numpy as np
import os, gc, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import vstack

print("="*55)
print("Truncation Sensitivity Analysis")
print("Char limits: 5k, 8k, 15k, 30k, full")
print("="*55)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

SOURCE_DIR = 'data/raw/Source Codes'
y   = df['quantum_broken'].values
idx = np.arange(len(df))
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)
rng = np.random.RandomState(42)

def read_src(row, limit):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        src = open(fpath,'r',errors='ignore').read()
        return src[:limit] if limit else src
    except:
        return ""

def build_matrix(indices, limit, label=""):
    hvec = HashingVectorizer(
        token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
        n_features=2**14, ngram_range=(1,2),
        alternate_sign=False, norm='l2')
    BATCH = 3000
    parts = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        texts = [read_src(df.iloc[i], limit)
                 for i in batch]
        parts.append(hvec.transform(texts))
        del texts; gc.collect()
    return vstack(parts)

def run(limit, label):
    print(f"\n  [{label}] Building matrices...")
    X_tr = build_matrix(tr_i, limit)
    pm = y_tr==1; nm = y_tr==0
    Xp = X_tr[pm]; Xn = X_tr[nm]
    yp = y_tr[pm]; yn = y_tr[nm]
    ui = rng.choice(len(yp),size=len(yn)//2,replace=True)
    Xb = vstack([Xn,Xp[ui]])
    yb = np.concatenate([yn,yp[ui]])
    del X_tr; gc.collect()
    X_te = build_matrix(te_i, limit)
    model = SGDClassifier(loss='modified_huber',
        class_weight='balanced',
        max_iter=200, random_state=42, n_jobs=-1)
    model.fit(Xb, yb)
    prob = model.predict_proba(X_te)[:,1]
    pred = model.predict(X_te)
    auc  = roc_auc_score(y_te, prob)
    f1   = f1_score(y_te, pred, zero_division=0)
    del X_te, Xb; gc.collect()
    print(f"  AUC={auc:.4f}  F1={f1:.4f}")
    return auc, f1

limits = [
    (5000,  "5k chars"),
    (8000,  "8k chars"),
    (15000, "15k chars"),
    (30000, "30k chars"),
    (None,  "Full file"),
]

results = []
for limit, label in limits:
    auc, f1 = run(limit, label)
    results.append({'limit':label,'AUC':auc,'F1':f1})

print("\n" + "="*45)
print("TRUNCATION SENSITIVITY RESULTS")
print("="*45)
print(f"{'Limit':<12}{'AUC':>8}{'F1':>8}")
print("-"*28)
for r in results:
    print(f"{r['limit']:<12}{r['AUC']:>8.4f}{r['F1']:>8.4f}")

pd.DataFrame(results).to_csv(
    'results/truncation_sensitivity.csv', index=False)
print("\nSaved: results/truncation_sensitivity.csv")
