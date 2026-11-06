"""
TDSC Upgrade: Out-of-Distribution Domain Generalization (LOCO)
Tests if the model learns dataset-specific templates or genuine structural features
by using Leave-One-Category-Out cross-validation.
"""
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

print("=" * 60)
print("Cross-Domain Generalization Experiment (LOCO)")
print("=" * 60)

labels = pd.read_csv('data/processed/quantum_labels.csv')
labels['quantum_broken'] = (
    (labels['Q1_ecrecover_detected'] == 1) |
    (labels['Q2_oz_ecdsa_detected']  == 1) |
    (labels['Q3_sig_checker_detected'] == 1)
).astype(int)

SRC_DIR = 'data/raw/Source Codes'
MAX_CHARS = 15000
TOKEN_PAT = r'[a-zA-Z_][a-zA-Z0-9_]*'

def load_source(row):
    for path in [
        os.path.join(SRC_DIR, row['category'], row['category'], row['filename']),
        os.path.join(SRC_DIR, row['category'], row['filename']),
    ]:
        if os.path.exists(path):
            return open(path, 'r', errors='ignore').read()[:MAX_CHARS]
    return ""

test_categories = ['Reentrancy', 'NonVulnerable', 'IntegerUO']
results = []

np.random.seed(42)

for test_cat in test_categories:
    print(f"\nEvaluating OOD Generalization on Domain: {test_cat}")
    
    # Split
    train_df = labels[labels['category'] != test_cat]
    test_df  = labels[labels['category'] == test_cat]
    
    # Subsample for RAM safety
    train_sub = train_df.sample(n=min(8000, len(train_df)), random_state=42)
    test_sub  = test_df.sample(n=min(1000, len(test_df)), random_state=42)
    
    print(f"  Training on {len(train_sub)} samples (11 categories)...")
    print(f"  Testing on {len(test_sub)} samples (Exclusively {test_cat})...")
    
    tr_texts = [load_source(row) for _, row in train_sub.iterrows()]
    te_texts = [load_source(row) for _, row in test_sub.iterrows()]
    
    # Filter empty
    tr_mask = [bool(t.strip()) for t in tr_texts]
    te_mask = [bool(t.strip()) for t in te_texts]
    
    tr_texts_f = [t for t, m in zip(tr_texts, tr_mask) if m]
    y_tr_f     = train_sub['quantum_broken'].values[tr_mask]
    
    te_texts_f = [t for t, m in zip(te_texts, te_mask) if m]
    y_te_f     = test_sub['quantum_broken'].values[te_mask]
    
    # Skip if test set doesn't have both classes
    if len(np.unique(y_te_f)) < 2:
        print(f"  Skipping {test_cat}: Insufficient class variance in test subset.")
        continue
        
    hv = HashingVectorizer(ngram_range=(1,2), n_features=2**14, token_pattern=TOKEN_PAT, norm='l2')
    X_tr = hv.transform(tr_texts_f)
    X_te = hv.transform(te_texts_f)
    
    clf = SGDClassifier(loss='modified_huber', random_state=42, max_iter=500)
    clf.fit(X_tr, y_tr_f)
    
    y_prob = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te_f, y_prob)
    print(f"  [+] OOD Generalization AUC: {auc:.4f}")
    
    results.append({
        'Test_Domain': test_cat,
        'Train_Size': len(y_tr_f),
        'Test_Size': len(y_te_f),
        'OOD_AUC': round(auc, 4)
    })

df_res = pd.DataFrame(results)
df_res.to_csv('results/ood_generalization.csv', index=False)
print("\n" + "=" * 60)
print(df_res.to_string(index=False))
print("=" * 60)
