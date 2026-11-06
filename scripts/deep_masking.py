import pandas as pd
import numpy as np
import os, re, warnings, gc
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix)
from scipy.sparse import vstack
from scipy import stats

print("="*60)
print("Deep Token Masking Experiment")
print("Three masking levels tested")
print("="*60)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

SOURCE_DIR = 'data/raw/Source Codes'

# Level 1: Function names only (original masking)
MASK_L1 = ['ecrecover','ECDSA','recover',
            'SignatureChecker','toEthSignedMessageHash']

# Level 2: Function names + parameter names
MASK_L2 = MASK_L1 + ['signature','sig','signer',
                      'signed','message','hash',
                      'digest','permit','nonce',
                      'deadline','domain','separator']

# Level 3: ALL ECDSA calling convention tokens
# Including parameter types used exclusively in ECDSA
MASK_L3 = MASK_L2 + ['v','r','s','bytes32','uint8',
                      'keccak256','mstore','calldataload',
                      'approve','meta','agent']

LEVELS = {
    'L1 (func names)':    MASK_L1,
    'L2 (+ param names)': MASK_L2,
    'L3 (+ all context)': MASK_L3,
}

y   = df['quantum_broken'].values
idx = np.arange(len(df))
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)
rng = np.random.RandomState(42)

def read_src(row, mask_tokens):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        src = open(fpath,'r',errors='ignore').read(15000)
    except:
        return ""
    for tok in mask_tokens:
        src = re.sub(r'\b'+re.escape(tok)+r'\b',
                     'MASKED', src)
    return src

hvec = HashingVectorizer(
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    n_features=2**14, ngram_range=(1,2),
    alternate_sign=False, norm='l2')

def build_matrix(indices, mask_tokens, label=""):
    BATCH = 3000
    parts = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        texts = [read_src(df.iloc[i], mask_tokens)
                 for i in batch]
        parts.append(hvec.transform(texts))
        del texts; gc.collect()
    return vstack(parts)

def run_exp(mask_tokens, label):
    X_tr = build_matrix(tr_i, mask_tokens)
    pm = y_tr==1; nm = y_tr==0
    Xp = X_tr[pm]; Xn = X_tr[nm]
    yp = y_tr[pm]; yn = y_tr[nm]
    ui = rng.choice(len(yp),size=len(yn)//2,replace=True)
    Xb = vstack([Xn,Xp[ui]])
    yb = np.concatenate([yn,yp[ui]])
    del X_tr; gc.collect()
    X_te = build_matrix(te_i, mask_tokens)
    model = SGDClassifier(loss='modified_huber',
        class_weight='balanced',
        max_iter=200, random_state=42, n_jobs=-1)
    model.fit(Xb, yb)
    prob = model.predict_proba(X_te)[:,1]
    pred = model.predict(X_te)
    auc  = roc_auc_score(y_te, prob)
    f1   = f1_score(y_te, pred, zero_division=0)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    del X_te, Xb; gc.collect()
    return auc, f1, prec, rec

# Baseline (no masking)
print("\n[0/3] BASELINE (no masking)...")
auc_base, f1_base, prec_base, rec_base = run_exp(
    [], "BASELINE")
print(f"  AUC={auc_base:.4f} F1={f1_base:.4f} "
      f"Prec={prec_base:.4f} Rec={rec_base:.4f}")

results = [{'Level':'Baseline (no mask)',
            'AUC':auc_base,'F1':f1_base,
            'Prec':prec_base,'Rec':rec_base}]

for i,(label,mask_toks) in enumerate(LEVELS.items(),1):
    print(f"\n[{i}/3] {label}...")
    auc,f1,prec,rec = run_exp(mask_toks, label)
    retention = auc/auc_base*100
    drop = (auc_base-auc)*100
    print(f"  AUC={auc:.4f} F1={f1:.4f} "
          f"Prec={prec:.4f} Rec={rec:.4f}")
    print(f"  Retention={retention:.1f}%  "
          f"Drop={drop:.2f}%")
    results.append({'Level':label,'AUC':auc,
                    'F1':f1,'Prec':prec,'Rec':rec,
                    'Retention_pct':round(retention,1),
                    'Drop_pct':round(drop,2)})

print("\n" + "="*60)
print("DEEP MASKING RESULTS")
print("="*60)
print(f"{'Level':<26}{'AUC':>8}{'F1':>7}"
      f"{'Ret%':>8}{'Drop%':>8}")
print("-"*57)
for r in results:
    ret = r.get('Retention_pct','—')
    drp = r.get('Drop_pct','—')
    print(f"{r['Level']:<26}{r['AUC']:>8.4f}"
          f"{r['F1']:>7.4f}{str(ret):>8}{str(drp):>8}")

pd.DataFrame(results).to_csv(
    'results/deep_masking.csv', index=False)
print("\nSaved: results/deep_masking.csv")
