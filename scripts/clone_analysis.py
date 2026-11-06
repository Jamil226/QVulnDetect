import pandas as pd
import numpy as np
import hashlib
import os
from tqdm import tqdm

print("="*55)
print("Clone/Duplicate Analysis")
print("="*55)

df = pd.read_csv('data/processed/quantum_labels.csv')
SOURCE_DIR = 'data/raw/Source Codes'

print(f"\nAnalysing {len(df):,} contracts for duplicates...")

# Step 1: Exact hash duplicates
hashes = []
for _, row in tqdm(df.iterrows(), total=len(df),
                   desc="Hashing"):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        src = open(fpath,'r',errors='ignore').read()
        # Normalise: strip comments and whitespace
        import re
        src_norm = re.sub(r'//.*?\n','',src)
        src_norm = re.sub(r'/\*.*?\*/','',
                          src_norm, flags=re.DOTALL)
        src_norm = re.sub(r'\s+','',src_norm)
        h = hashlib.md5(src_norm.encode()).hexdigest()
    except:
        h = "error_" + str(_)
    hashes.append(h)

df['content_hash'] = hashes

# Count exact duplicates
total        = len(df)
unique_hashes= df['content_hash'].nunique()
duplicates   = total - unique_hashes
dup_pct      = duplicates/total*100

print(f"\nTotal contracts    : {total:,}")
print(f"Unique (by hash)   : {unique_hashes:,}")
print(f"Exact duplicates   : {duplicates:,} ({dup_pct:.1f}%)")

# Step 2: Check if duplicates straddle train/test
from sklearn.model_selection import train_test_split
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

y   = df['quantum_broken'].values
idx = np.arange(total)
tr_i, te_i, _, _ = train_test_split(
    idx, y, test_size=0.2,
    random_state=42, stratify=y)

train_hashes = set(df.iloc[tr_i]['content_hash'])
test_hashes  = set(df.iloc[te_i]['content_hash'])
overlap      = train_hashes & test_hashes
overlap_pct  = len(overlap)/len(test_hashes)*100

print(f"\nTrain unique hashes: {len(train_hashes):,}")
print(f"Test unique hashes : {len(test_hashes):,}")
print(f"Overlapping hashes : {len(overlap):,}")
print(f"Test contracts with train clone: "
      f"{overlap_pct:.1f}%")

# Step 3: Deduplicated dataset stats
df_dedup = df.drop_duplicates(subset='content_hash')
qb_dedup = df_dedup['quantum_broken'].sum()
print(f"\nAfter deduplication:")
print(f"  Unique contracts : {len(df_dedup):,}")
print(f"  Quantum-broken   : {qb_dedup:,} "
      f"({qb_dedup/len(df_dedup)*100:.1f}%)")

# Save
df['content_hash'].to_csv(
    'results/contract_hashes.csv', index=False)
pd.DataFrame({
    'metric': ['total','unique','duplicates',
               'dup_pct','overlap_pct',
               'dedup_total','dedup_qb'],
    'value':  [total, unique_hashes, duplicates,
               round(dup_pct,1), round(overlap_pct,1),
               len(df_dedup), int(qb_dedup)]
}).to_csv('results/clone_analysis.csv', index=False)
print("\nSaved: results/clone_analysis.csv")
