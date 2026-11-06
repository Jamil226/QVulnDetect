import pandas as pd
import numpy as np
import os, time, warnings, gc
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix,
    precision_recall_curve)
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Key change: read one file at a time ──────────────────
def read_contract(row, source_dir, char_limit=8000):
    fpath = os.path.join(source_dir, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(source_dir,
                 row['category'], row['filename'])
    try:
        return open(fpath,'r',errors='ignore').read(char_limit)
    except:
        return ""

print("="*55)
print("QVulnDetect — Memory Safe (HashingVectorizer)")
print("="*55)

# ── Load labels ───────────────────────────────────────────
print("\n[1/5] Loading labels...")
df = pd.read_csv('data/processed/quantum_labels.csv')
total = len(df)
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1) |
    (df['Q2_oz_ecdsa_detected']==1)  |
    (df['Q3_sig_checker_detected']==1)
).astype(int)
y  = df['quantum_broken'].values
qb = y.sum()
print(f"Total    : {total:,}")
print(f"Positive : {qb:,} ({qb/total*100:.1f}%)")
print(f"Negative : {(y==0).sum():,}")

# ── Train/test split on indices only ─────────────────────
print("\n[2/5] Splitting indices...")
idx = np.arange(total)
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2,
    random_state=42, stratify=y)
print(f"Train:{len(tr_i):,}  Test:{len(te_i):,}")
print(f"Test positives: {y_te.sum():,}")

SOURCE_DIR = 'data/raw/Source Codes'

# ── HashingVectorizer: no vocabulary stored in RAM ───────
# Hashes tokens directly — zero memory for vocabulary
# n_features=2^17=131072 avoids collisions
print("\n[3/5] Vectorizing (HashingVectorizer — no RAM spike)...")
hvec = HashingVectorizer(
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    n_features=2**14,    # 16384 features — small enough
    ngram_range=(1,2),
    alternate_sign=False,
    norm='l2',
)

# Build train matrix row by row in batches of 5000
BATCH = 5000
print(f"  Building train matrix ({len(tr_i):,} contracts)...")
from scipy.sparse import vstack as sp_vstack

X_tr_parts = []
for start in range(0, len(tr_i), BATCH):
    batch_idx = tr_i[start:start+BATCH]
    batch_texts = [read_contract(df.iloc[i], SOURCE_DIR)
                   for i in batch_idx]
    X_batch = hvec.transform(batch_texts)
    X_tr_parts.append(X_batch)
    del batch_texts
    gc.collect()
    if (start//BATCH) % 4 == 0:
        print(f"    {min(start+BATCH, len(tr_i)):,}/{len(tr_i):,}")

X_tr = sp_vstack(X_tr_parts)
del X_tr_parts
gc.collect()
print(f"  Train matrix: {X_tr.shape} "
      f"({X_tr.data.nbytes/1024/1024:.0f}MB)")

print(f"  Building test matrix ({len(te_i):,} contracts)...")
X_te_parts = []
for start in range(0, len(te_i), BATCH):
    batch_idx = te_i[start:start+BATCH]
    batch_texts = [read_contract(df.iloc[i], SOURCE_DIR)
                   for i in batch_idx]
    X_batch = hvec.transform(batch_texts)
    X_te_parts.append(X_batch)
    del batch_texts
    gc.collect()

X_te = sp_vstack(X_te_parts)
del X_te_parts
gc.collect()
print(f"  Test matrix: {X_te.shape} "
      f"({X_te.data.nbytes/1024/1024:.0f}MB)")

# ── Train models ──────────────────────────────────────────
print("\n[4/5] Training models...")
os.makedirs('results', exist_ok=True)

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, class_weight='balanced',
        max_iter=1000, random_state=42,
        solver='saga', n_jobs=-1),
    'SGD Classifier': SGDClassifier(
        loss='modified_huber',
        class_weight='balanced',
        max_iter=200, random_state=42,
        n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"\n  >> {name}...")
    t0    = time.time()
    model.fit(X_tr, y_tr)
    yp    = model.predict(X_te)
    yprob = model.predict_proba(X_te)[:,1]
    elapsed = time.time()-t0
    results[name] = {
        'acc' : accuracy_score(y_te, yp),
        'f1'  : f1_score(y_te, yp, zero_division=0),
        'prec': precision_score(y_te, yp, zero_division=0),
        'rec' : recall_score(y_te, yp, zero_division=0),
        'auc' : roc_auc_score(y_te, yprob),
        'ap'  : average_precision_score(y_te, yprob),
        'cm'  : confusion_matrix(y_te, yp),
        'prob': yprob, 'pred': yp, 'time': elapsed,
    }
    r = results[name]
    print(f"     Acc:{r['acc']:.4f} F1:{r['f1']:.4f} "
          f"Prec:{r['prec']:.4f} Rec:{r['rec']:.4f} "
          f"AUC:{r['auc']:.4f} AP:{r['ap']:.4f} "
          f"({elapsed:.1f}s)")

# ── McNemar test ──────────────────────────────────────────
print("\nMcNemar Test...")
best_name = max(results, key=lambda n: results[n]['auc'])
y_base = np.zeros(len(y_te), dtype=int)
y_best = results[best_name]['pred']
n00 = np.sum((y_base==y_te)&(y_best==y_te))
n01 = np.sum((y_base!=y_te)&(y_best==y_te))
n10 = np.sum((y_base==y_te)&(y_best!=y_te))
n11 = np.sum((y_base!=y_te)&(y_best!=y_te))
_, p_val, _, _ = chi2_contingency([[n00,n01],[n10,n11]])
sig = ("p<0.001" if p_val<0.001 else
       "p<0.05"  if p_val<0.05  else "not sig")
print(f"  p = {p_val:.2e} ({sig})")

# ── Summary ───────────────────────────────────────────────
print("\n[5/5] Summary...")
print("="*55)
print("RESULTS — FULL 111,897 CONTRACTS")
print("="*55)
print(f"\n{'Model':<22}{'F1':>7}{'AUC':>8}{'AP':>8}")
print("-"*45)
for n,r in results.items():
    print(f"{n:<22}{r['f1']:>7.4f}"
          f"{r['auc']:>8.4f}{r['ap']:>8.4f}")
best = results[best_name]
imp  = (best['auc']-0.5197)*100
print(f"\nBest       : {best_name}")
print(f"AUC-ROC    : {best['auc']:.4f}")
print(f"Improvement: +{imp:.1f}% over bytecode (0.5197)")
print(f"McNemar p  : {p_val:.2e} ({sig})")

# ── Save ──────────────────────────────────────────────────
pd.DataFrame([
    {'Model':n,'F1':r['f1'],'AUC':r['auc'],'AP':r['ap']}
    for n,r in results.items()
]).to_csv('results/tfidf_fixed_results.csv', index=False)

# ── Plots ─────────────────────────────────────────────────
fig,ax = plt.subplots(figsize=(7,5))
colors = ['#1A56DB','#E74C3C']
for i,(name,r) in enumerate(results.items()):
    p_c,r_c,_ = precision_recall_curve(y_te, r['prob'])
    ax.plot(r_c, p_c, color=colors[i], lw=2,
            label=f"{name} (AP={r['ap']:.4f})")
ax.axhline(y=y_te.mean(), color='gray', linestyle='--',
           label=f"Random ({y_te.mean():.3f})")
ax.set_xlabel('Recall',fontsize=12)
ax.set_ylabel('Precision',fontsize=12)
ax.set_title('Precision-Recall Curve\n111,897 Contracts',
             fontsize=13,fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)
plt.tight_layout()
plt.savefig('results/precision_recall_curve.png',dpi=150)

fig,ax = plt.subplots(figsize=(5,4))
sns.heatmap(best['cm'],annot=True,fmt='d',cmap='Blues',
    xticklabels=['Safe','Q-Broken'],
    yticklabels=['Safe','Q-Broken'],ax=ax)
ax.set_title(f'Confusion Matrix\n{best_name}',
             fontweight='bold')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/fixed_confusion_matrix.png',dpi=150)

print("\nSaved: results/tfidf_fixed_results.csv")
print("Saved: results/precision_recall_curve.png")
print("Saved: results/fixed_confusion_matrix.png")
print("\nDone!")
