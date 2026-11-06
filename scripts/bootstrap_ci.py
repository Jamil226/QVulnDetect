import pandas as pd
import numpy as np
import os, warnings, gc
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from scipy.sparse import vstack

print("="*55)
print("Bootstrap AUC Confidence Intervals")
print("More reliable than Hanley-McNeil at AUC~0.99")
print("="*55)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

SOURCE_DIR = 'data/raw/Source Codes'
y   = df['quantum_broken'].values
idx = np.arange(len(df))

# Use same split
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)

def read_src(row):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        return open(fpath,'r',errors='ignore').read(15000)
    except:
        return ""

hvec = HashingVectorizer(
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    n_features=2**14, ngram_range=(1,2),
    alternate_sign=False, norm='l2')

BATCH = 3000
rng   = np.random.RandomState(42)

def build_matrix(indices):
    parts = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        texts = [read_src(df.iloc[i]) for i in batch]
        parts.append(hvec.transform(texts))
        del texts; gc.collect()
    return vstack(parts)

print("\n[1/3] Building matrices...")
X_tr = build_matrix(tr_i)
pm = y_tr==1; nm = y_tr==0
Xp = X_tr[pm]; Xn = X_tr[nm]
yp = y_tr[pm]; yn = y_tr[nm]
ui = rng.choice(len(yp),size=len(yn)//2,replace=True)
Xb = vstack([Xn,Xp[ui]]); yb = np.concatenate([yn,yp[ui]])
del X_tr; gc.collect()
X_te = build_matrix(te_i)

print("[2/3] Training SGD...")
model = SGDClassifier(loss='modified_huber',
    class_weight='balanced',
    max_iter=200, random_state=42, n_jobs=-1)
model.fit(Xb, yb)
y_prob = model.predict_proba(X_te)[:,1]
auc_obs = roc_auc_score(y_te, y_prob)
print(f"  Observed AUC: {auc_obs:.4f}")

print("[3/3] Bootstrap CI (1000 iterations)...")
N_BOOT  = 1000
boot_aucs = []
rng2 = np.random.RandomState(123)
n_te = len(y_te)

for i in range(N_BOOT):
    boot_idx = rng2.choice(n_te, size=n_te, replace=True)
    y_te_b   = y_te[boot_idx]
    y_pr_b   = y_prob[boot_idx]
    if y_te_b.sum() == 0 or y_te_b.sum() == n_te:
        continue
    try:
        boot_aucs.append(roc_auc_score(y_te_b, y_pr_b))
    except:
        pass

boot_aucs = np.array(boot_aucs)
ci_low  = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)
ci_mean = np.mean(boot_aucs)
ci_std  = np.std(boot_aucs)

print(f"\n{'='*50}")
print(f"BOOTSTRAP AUC RESULTS (n_boot={len(boot_aucs)})")
print(f"{'='*50}")
print(f"Observed AUC   : {auc_obs:.4f}")
print(f"Bootstrap mean : {ci_mean:.4f}")
print(f"Bootstrap std  : {ci_std:.4f}")
print(f"95% CI (perc.) : [{ci_low:.4f}, {ci_high:.4f}]")
print(f"99% CI (perc.) : "
      f"[{np.percentile(boot_aucs,0.5):.4f}, "
      f"{np.percentile(boot_aucs,99.5):.4f}]")

# Also run with seed=123 for stability check
print("\n[EXTRA] Second seed (seed=123) for stability...")
tr2, te2, ytr2, yte2 = train_test_split(
    idx, y, test_size=0.2, random_state=123, stratify=y)
X_tr2 = build_matrix(tr2)
pm2 = ytr2==1; nm2 = ytr2==0
Xp2 = X_tr2[pm2]; Xn2 = X_tr2[nm2]
yp2 = ytr2[pm2]; yn2 = ytr2[nm2]
rng3 = np.random.RandomState(123)
ui2  = rng3.choice(len(yp2),size=len(yn2)//2,replace=True)
Xb2 = vstack([Xn2,Xp2[ui2]])
yb2 = np.concatenate([yn2,yp2[ui2]])
del X_tr2; gc.collect()
X_te2 = build_matrix(te2)
m2 = SGDClassifier(loss='modified_huber',
    class_weight='balanced',
    max_iter=200, random_state=123, n_jobs=-1)
m2.fit(Xb2, yb2)
auc2 = roc_auc_score(yte2, m2.predict_proba(X_te2)[:,1])
f1_2 = f1_score(yte2, m2.predict(X_te2), zero_division=0)
print(f"  Seed 123 AUC : {auc2:.4f}")
print(f"  Seed 123 F1  : {f1_2:.4f}")
print(f"  Seed 42  AUC : {auc_obs:.4f}")
print(f"  Difference   : {abs(auc_obs-auc2):.4f}")
if abs(auc_obs-auc2) < 0.01:
    print(f"  => Results are STABLE across seeds")

pd.DataFrame({
    'metric':['observed_auc','boot_mean','boot_std',
              'ci95_low','ci95_high','seed123_auc'],
    'value': [auc_obs, ci_mean, ci_std,
              ci_low, ci_high, auc2]
}).to_csv('results/bootstrap_ci.csv', index=False)
print("\nSaved: results/bootstrap_ci.csv")
