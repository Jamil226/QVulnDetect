import pandas as pd
import numpy as np
import os, warnings, gc
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    average_precision_score)
from scipy.sparse import vstack
from scipy import stats

print("="*60)
print("Deduplicated Test Set Evaluation")
print("Removes all test contracts with train duplicate")
print("="*60)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

# Load precomputed hashes
hashes = pd.read_csv('results/contract_hashes.csv')
df['content_hash'] = hashes['content_hash'].values

y   = df['quantum_broken'].values
idx = np.arange(len(df))
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)

# Find test contracts with no train duplicate
train_hashes = set(df.iloc[tr_i]['content_hash'].values)
te_unique_mask = np.array([
    df.iloc[i]['content_hash'] not in train_hashes
    for i in te_i])

te_i_dedup  = te_i[te_unique_mask]
y_te_dedup  = y_te[te_unique_mask]

print(f"\nOriginal test set    : {len(te_i):,} contracts")
print(f"Test with train clone: "
      f"{(~te_unique_mask).sum():,} "
      f"({(~te_unique_mask).mean()*100:.1f}%)")
print(f"Deduplicated test    : {len(te_i_dedup):,} contracts")
print(f"Dedup test positives : {y_te_dedup.sum():,} "
      f"({y_te_dedup.mean()*100:.1f}%)")

if len(te_i_dedup) < 100:
    print("ERROR: Too few contracts after dedup")
    exit()

SOURCE_DIR = 'data/raw/Source Codes'
rng = np.random.RandomState(42)

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

def build_matrix(indices):
    parts = []
    for start in range(0, len(indices), BATCH):
        batch = indices[start:start+BATCH]
        texts = [read_src(df.iloc[i]) for i in batch]
        parts.append(hvec.transform(texts))
        del texts; gc.collect()
        print(f"  {min(start+BATCH,len(indices)):,}/"
              f"{len(indices):,}")
    return vstack(parts)

print("\n[1/3] Building train matrix...")
X_tr = build_matrix(tr_i)
pm = y_tr==1; nm = y_tr==0
Xp = X_tr[pm]; Xn = X_tr[nm]
yp = y_tr[pm]; yn = y_tr[nm]
ui = rng.choice(len(yp),size=len(yn)//2,replace=True)
Xb = vstack([Xn,Xp[ui]])
yb = np.concatenate([yn,yp[ui]])
del X_tr; gc.collect()

print("\n[2/3] Building full test matrix...")
X_te_full = build_matrix(te_i)

print("\n[3/3] Building dedup test matrix...")
X_te_dedup = build_matrix(te_i_dedup)

print("\nTraining SGD...")
model = SGDClassifier(loss='modified_huber',
    class_weight='balanced',
    max_iter=200, random_state=42, n_jobs=-1)
model.fit(Xb, yb)

# Full test evaluation
prob_f = model.predict_proba(X_te_full)[:,1]
pred_f = model.predict(X_te_full)
auc_f  = roc_auc_score(y_te, prob_f)
f1_f   = f1_score(y_te, pred_f, zero_division=0)
prec_f = precision_score(y_te, pred_f, zero_division=0)
rec_f  = recall_score(y_te, pred_f, zero_division=0)
ap_f   = average_precision_score(y_te, prob_f)
cm_f   = confusion_matrix(y_te, pred_f)

# Dedup test evaluation
prob_d = model.predict_proba(X_te_dedup)[:,1]
pred_d = model.predict(X_te_dedup)
auc_d  = roc_auc_score(y_te_dedup, prob_d)
f1_d   = f1_score(y_te_dedup, pred_d, zero_division=0)
prec_d = precision_score(y_te_dedup, pred_d, zero_division=0)
rec_d  = recall_score(y_te_dedup, pred_d, zero_division=0)
ap_d   = average_precision_score(y_te_dedup, prob_d)
cm_d   = confusion_matrix(y_te_dedup, pred_d)

print("\n" + "="*60)
print("DEDUPLICATION EVALUATION RESULTS")
print("="*60)
print(f"\n{'Metric':<20}{'Full Test':>12}"
      f"{'Dedup Test':>12}{'Difference':>12}")
print("-"*56)
print(f"{'n_test':<20}{len(te_i):>12,}"
      f"{len(te_i_dedup):>12,}")
print(f"{'n_positive':<20}{y_te.sum():>12,}"
      f"{y_te_dedup.sum():>12,}")
print(f"{'AUC':<20}{auc_f:>12.4f}{auc_d:>12.4f}"
      f"{auc_d-auc_f:>+12.4f}")
print(f"{'F1':<20}{f1_f:>12.4f}{f1_d:>12.4f}"
      f"{f1_d-f1_f:>+12.4f}")
print(f"{'Precision':<20}{prec_f:>12.4f}{prec_d:>12.4f}"
      f"{prec_d-prec_f:>+12.4f}")
print(f"{'Recall':<20}{rec_f:>12.4f}{rec_d:>12.4f}"
      f"{rec_d-rec_f:>+12.4f}")
print(f"{'Avg Precision':<20}{ap_f:>12.4f}{ap_d:>12.4f}"
      f"{ap_d-ap_f:>+12.4f}")

print(f"\nFull test confusion matrix:")
print(f"  TN={cm_f[0,0]:,}  FP={cm_f[0,1]:,}")
print(f"  FN={cm_f[1,0]:,}  TP={cm_f[1,1]:,}")

print(f"\nDedup test confusion matrix:")
print(f"  TN={cm_d[0,0]:,}  FP={cm_d[0,1]:,}")
print(f"  FN={cm_d[1,0]:,}  TP={cm_d[1,1]:,}")

# Key conclusion
diff = auc_f - auc_d
print(f"\nKEY FINDING:")
print(f"AUC drop from full to dedup: {diff:+.4f}")
if abs(diff) < 0.02:
    print("=> Minimal AUC change after deduplication.")
    print("=> Clone contamination has limited impact on")
    print("   reported performance for this task.")
elif diff > 0.02:
    print("=> AUC drops after deduplication.")
    print("=> Clone contamination inflates full-set results.")
    print("=> Deduplicated AUC is the more reliable metric.")

pd.DataFrame([
    {'Eval':'Full Test','n':len(te_i),'n_pos':y_te.sum(),
     'AUC':auc_f,'F1':f1_f,'Prec':prec_f,
     'Rec':rec_f,'AP':ap_f},
    {'Eval':'Dedup Test','n':len(te_i_dedup),
     'n_pos':y_te_dedup.sum(),
     'AUC':auc_d,'F1':f1_d,'Prec':prec_d,
     'Rec':rec_d,'AP':ap_d},
]).to_csv('results/dedup_evaluation.csv', index=False)
print("\nSaved: results/dedup_evaluation.csv")
