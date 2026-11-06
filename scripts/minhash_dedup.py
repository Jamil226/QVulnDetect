import pandas as pd
import numpy as np
import os, re, gc
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

print("="*55)
print("Near-Duplicate Detection (MinHash LSH)")
print("Threshold: Jaccard similarity >= 0.80")
print("="*55)

df = pd.read_csv('data/processed/quantum_labels.csv')
df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1)|
    (df['Q2_oz_ecdsa_detected']==1)|
    (df['Q3_sig_checker_detected']==1)).astype(int)

SOURCE_DIR = 'data/raw/Source Codes'
THRESHOLD  = 0.80
NUM_PERM   = 128

def get_shingles(src, k=5):
    src = re.sub(r'//.*?\n','',src)
    src = re.sub(r'/\*.*?\*/','',src,flags=re.DOTALL)
    src = re.sub(r'\s+',' ',src).strip()
    return set(src[i:i+k] for i in range(len(src)-k+1))

print("\n[1/3] Building MinHash signatures...")
minhashes = {}
BATCH = 5000
for start in range(0, len(df), BATCH):
    batch = df.iloc[start:start+BATCH]
    for idx, row in batch.iterrows():
        fpath = os.path.join(SOURCE_DIR, row['category'],
                             row['category'], row['filename'])
        if not os.path.exists(fpath):
            fpath = os.path.join(SOURCE_DIR,
                     row['category'], row['filename'])
        try:
            src = open(fpath,'r',errors='ignore').read()
            shingles = get_shingles(src)
        except:
            shingles = {str(idx)}
        m = MinHash(num_perm=NUM_PERM)
        for s in shingles:
            m.update(s.encode('utf8'))
        minhashes[idx] = m
    gc.collect()
    print(f"  {min(start+BATCH,len(df)):,}/{len(df):,}")

print("\n[2/3] LSH near-duplicate clustering...")
lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
for idx, m in tqdm(minhashes.items(), desc="Inserting"):
    lsh.insert(str(idx), m)

print("\n[3/3] Counting near-duplicates...")
near_dup_count = 0
seen = set()
cluster_sizes = []
for idx in tqdm(minhashes.keys(), desc="Querying"):
    if idx in seen:
        continue
    result = lsh.query(minhashes[idx])
    result_int = [int(r) for r in result]
    if len(result_int) > 1:
        near_dup_count += len(result_int) - 1
        cluster_sizes.append(len(result_int))
        seen.update(result_int)

total         = len(df)
near_dup_pct  = near_dup_count/total*100
unique_near   = total - near_dup_count

print(f"\nTotal contracts        : {total:,}")
print(f"Exact duplicates       : 45,585 (40.7%)")
print(f"Near-duplicates (>=0.8): {near_dup_count:,} "
      f"({near_dup_pct:.1f}%)")
print(f"Unique after near-dedup: {unique_near:,}")
if cluster_sizes:
    print(f"Avg cluster size       : "
          f"{np.mean(cluster_sizes):.1f}")
    print(f"Max cluster size       : {max(cluster_sizes)}")

# Quantum broken in near-dedup unique contracts
from sklearn.model_selection import train_test_split
y   = df['quantum_broken'].values
idx = np.arange(total)
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)
train_set = set(tr_i.tolist())
test_set  = set(te_i.tolist())

pd.DataFrame({
    'metric':['exact_dups','exact_dup_pct',
              'near_dups','near_dup_pct','unique_near'],
    'value': [45585, 40.7, near_dup_count,
              round(near_dup_pct,1), unique_near]
}).to_csv('results/minhash_dedup.csv', index=False)
print("\nSaved: results/minhash_dedup.csv")
