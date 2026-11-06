import pandas as pd
import numpy as np
import os, re, warnings, gc
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import vstack

print("="*55)
print("Token Masking — Memory Safe Version")
print("="*55)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

SOURCE_DIR = 'data/raw/Source Codes'
MASK_TOKENS = ['ecrecover','ECDSA','recover',
               'SignatureChecker','toEthSignedMessageHash']

y   = df['quantum_broken'].values
idx = np.arange(len(df))
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)

rng = np.random.RandomState(42)

def read_src(row, mask=False):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        src = open(fpath,'r',errors='ignore').read(15000)
    except:
        return ""
    if mask:
        for tok in MASK_TOKENS:
            src = re.sub(r'\b'+tok+r'\b','MASKED',src)
    return src

hvec = HashingVectorizer(
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    n_features=2**14, ngram_range=(1,2),
    alternate_sign=False, norm='l2')

def build_matrix(indices, mask=False, label=""):
    BATCH = 3000
    parts = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        texts = [read_src(df.iloc[i], mask=mask)
                 for i in batch]
        parts.append(hvec.transform(texts))
        del texts; gc.collect()
        done = min(start+BATCH, len(indices))
        print(f"    {label}: {done:,}/{len(indices):,}")
    return vstack(parts)

def oversample(X, y):
    pm = y==1; nm = y==0
    Xp = X[pm]; Xn = X[nm]
    yp = y[pm]; yn = y[nm]
    ui = rng.choice(len(yp),
                    size=len(yn)//2, replace=True)
    Xb = vstack([Xn, Xp[ui]])
    yb = np.concatenate([yn, yp[ui]])
    return Xb, yb

def run_exp(mask, label):
    print(f"\n  Building train matrix ({label})...")
    X_tr = build_matrix(tr_i, mask=mask, label="train")
    Xb, yb = oversample(X_tr, y_tr)
    del X_tr; gc.collect()

    print(f"  Building test matrix ({label})...")
    X_te = build_matrix(te_i, mask=mask, label="test")

    model = SGDClassifier(
        loss='modified_huber', class_weight='balanced',
        max_iter=200, random_state=42, n_jobs=-1)
    model.fit(Xb, yb)
    prob = model.predict_proba(X_te)[:,1]
    pred = model.predict(X_te)
    auc  = roc_auc_score(y_te, prob)
    f1   = f1_score(y_te, pred, zero_division=0)
    del X_te, Xb; gc.collect()
    return auc, f1

print("\n[1/2] NORMAL text (no masking)...")
auc_n, f1_n = run_exp(mask=False, label="NORMAL")
print(f"  AUC={auc_n:.4f}  F1={f1_n:.4f}")

print("\n[2/2] MASKED text (ECDSA tokens removed)...")
auc_m, f1_m = run_exp(mask=True, label="MASKED")
print(f"  AUC={auc_m:.4f}  F1={f1_m:.4f}")

print("\n" + "="*55)
print("MASKING RESULTS")
print("="*55)
retention = auc_m/auc_n*100
drop      = (auc_n-auc_m)*100
print(f"Normal AUC  : {auc_n:.4f}")
print(f"Masked AUC  : {auc_m:.4f}")
print(f"Retention   : {retention:.1f}%")
print(f"AUC drop    : {drop:.2f}%")

if auc_m > 0.85:
    verdict = ("STRONG semantic learning — model "
               "classifies correctly without any "
               "explicit ECDSA token names")
elif auc_m > 0.70:
    verdict = ("PARTIAL semantic learning — some "
               "context signal retained after masking")
else:
    verdict = ("WEAK — model relies heavily on "
               "explicit token names")
print(f"Verdict     : {verdict}")

pd.DataFrame([
    {'Experiment':'Normal','AUC':auc_n,'F1':f1_n},
    {'Experiment':'Masked','AUC':auc_m,'F1':f1_m},
]).to_csv('results/token_masking.csv', index=False)
print("\nSaved: results/token_masking.csv")
