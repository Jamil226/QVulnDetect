import pandas as pd
import numpy as np
import os, gc, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import vstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("="*55)
print("Feature Importance — LR Coefficient Analysis")
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

# Use TfidfVectorizer on TRAIN only for interpretability
# (HashingVectorizer has no inverse transform)
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n[1/3] Loading train source files...")
BATCH = 3000
tr_texts = []
for start in range(0, len(tr_i), BATCH):
    batch = tr_i[start:start+BATCH]
    texts = []
    for i in batch:
        row   = df.iloc[i]
        fpath = os.path.join(SOURCE_DIR, row['category'],
                             row['category'], row['filename'])
        if not os.path.exists(fpath):
            fpath = os.path.join(SOURCE_DIR,
                     row['category'], row['filename'])
        try:
            texts.append(
                open(fpath,'r',errors='ignore').read(15000))
        except:
            texts.append("")
    tr_texts.extend(texts)
    del texts; gc.collect()
    print(f"  {min(start+BATCH,len(tr_i)):,}/{len(tr_i):,}")

print("\n[2/3] Fitting TF-IDF on train (for interpretability)...")
vec = TfidfVectorizer(
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    max_features=3000, min_df=5, max_df=0.95,
    ngram_range=(1,2), sublinear_tf=True)
X_tr = vec.fit_transform(tr_texts)
del tr_texts; gc.collect()

# Oversample
rng = np.random.RandomState(42)
pm  = y_tr==1; nm = y_tr==0
Xp  = X_tr[pm]; Xn = X_tr[nm]
yp  = y_tr[pm]; yn = y_tr[nm]
ui  = rng.choice(len(yp),size=len(yn)//2,replace=True)
Xb  = vstack([Xn,Xp[ui]])
yb  = np.concatenate([yn,yp[ui]])

print("\n[3/3] Training LR for coefficient analysis...")
lr = LogisticRegression(C=1.0, class_weight='balanced',
         max_iter=1000, random_state=42, solver='saga')
lr.fit(Xb, yb)

vocab  = vec.get_feature_names_out()
coefs  = lr.coef_[0]

# Top 30 positive (quantum-vulnerable) tokens
top30_pos_idx = np.argsort(coefs)[-30:][::-1]
top30_neg_idx = np.argsort(coefs)[:15]

print("\nTop 30 tokens → QUANTUM VULNERABLE:")
print(f"{'Rank':<5}{'Token':<30}{'Coef':>9}{'Semantic Role'}")
print("-"*70)

semantic = {
    'bytes32':     'Hash/sig param type in ecrecover',
    'v':           'Recovery param: ecrecover(h,v,r,s)',
    'uint8':       'Type of v parameter',
    'r':           'Sig param: ecrecover(h,v,r,s)',
    'signature':   'ECDSA signing variable',
    'keccak256':   'Pre-signing hash function',
    'message':     'Signed message variable',
    'signer':      'Recovered signer address',
    'hash':        'Message hash variable',
    'recover':     'Generic recovery pattern',
    'sig':         'Signature abbreviation',
    'permit':      'ERC-2612 permit pattern',
    'nonce':       'Replay protection in sig verify',
    'deadline':    'Sig expiry in permit',
    'digest':      'EIP-712 structured hash',
    'domain':      'EIP-712 domain separator',
    'separator':   'EIP-712 domain separator',
    'signed':      'Pre-signed message pattern',
    'approve':     'Approval via signature',
    'meta':        'Meta-transaction pattern',
}

rows = []
for rank, i in enumerate(top30_pos_idx, 1):
    tok  = vocab[i]
    role = semantic.get(tok, 'ECDSA context token')
    print(f"{rank:<5}{tok:<30}{coefs[i]:>9.4f}  {role}")
    rows.append({'rank':rank,'token':tok,
                 'coefficient':round(coefs[i],4),
                 'direction':'vulnerable','role':role})

print("\nTop 15 tokens → QUANTUM SAFE:")
for rank, i in enumerate(top30_neg_idx, 1):
    tok = vocab[i]
    print(f"{rank:<5}{tok:<30}{coefs[i]:>9.4f}")
    rows.append({'rank':rank,'token':tok,
                 'coefficient':round(coefs[i],4),
                 'direction':'safe','role':'safe pattern'})

pd.DataFrame(rows).to_csv(
    'results/feature_importance.csv', index=False)

# Plot top 20 positive + top 10 negative
fig, ax = plt.subplots(figsize=(10, 7))
top20_idx = np.argsort(coefs)[-20:]
top10_neg = np.argsort(coefs)[:10]
plot_idx  = np.concatenate([top10_neg, top20_idx])
plot_vals = coefs[plot_idx]
plot_labs = vocab[plot_idx]
colors    = ['#E74C3C' if v>0 else '#3498DB'
             for v in plot_vals]
bars = ax.barh(range(len(plot_idx)),
               plot_vals, color=colors, alpha=0.85)
ax.set_yticks(range(len(plot_idx)))
ax.set_yticklabels(plot_labs, fontsize=9)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Logistic Regression Coefficient',
              fontsize=12)
ax.set_title('Feature Importance — Top Predictive Tokens\n'
             'Red: Quantum-Vulnerable  Blue: Quantum-Safe',
             fontsize=13, fontweight='bold')
red_p = mpatches.Patch(color='#E74C3C',
                        label='Quantum-Vulnerable')
blu_p = mpatches.Patch(color='#3498DB',
                        label='Quantum-Safe')
ax.legend(handles=[red_p, blu_p], fontsize=10)
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
print("\nSaved: results/feature_importance.csv")
print("Saved: results/feature_importance.png")
