import pandas as pd
import os
import re
from tqdm import tqdm

PATTERNS = {
    'Q1_ecrecover':       (r'ecrecover\s*\(',         'CRITICAL'),
    'Q2_oz_ecdsa':        (r'ECDSA\.recover\s*\(',    'CRITICAL'),
    'Q3_sig_checker':     (r'SignatureChecker\.',      'CRITICAL'),
    'Q4_eth_signed_hash': (r'toEthSignedMessageHash', 'HIGH'),
    'Q5_timestamp_prng':  (r'block\.timestamp',       'HIGH'),
    'Q6_blockhash_prng':  (r'blockhash\s*\(',         'HIGH'),
    'Q7_sha256_auth':     (r'sha256\s*\(',            'MEDIUM'),
    'Q8_secp256k1':       (r'secp256k1',              'MEDIUM'),
    'Q9_pubkey_storage':  (r'publicKey|pubKey|ecKey', 'MEDIUM'),
}

SEVERITY_WEIGHT = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1}
SOURCE_DIR = 'data/raw/Source Codes'

# ── Collect all .sol files ───────────────────────────────
print("Scanning source code files...")
sol_files = []

for category in os.listdir(SOURCE_DIR):
    cat_path = os.path.join(SOURCE_DIR, category)
    if not os.path.isdir(cat_path) or cat_path.endswith('.zip'):
        continue
    # Walk ALL subdirectories (handles nested structure)
    for root, dirs, files in os.walk(cat_path):
        for fname in files:
            if fname.endswith('.sol'):
                sol_files.append({
                    'filename': fname,
                    'category': category,
                    'filepath': os.path.join(root, fname)
                })

print(f"Found {len(sol_files):,} .sol files")
print(f"Categories: {sorted(set(f['category'] for f in sol_files))}")

# ── Label each contract ──────────────────────────────────
results = []
for item in tqdm(sol_files, desc="Labeling"):
    try:
        with open(item['filepath'], 'r', errors='ignore') as f:
            source = f.read()
    except:
        source = ""

    row = {
        'filename': item['filename'],
        'category': item['category'],
    }

    risk_score   = 0
    any_critical = False

    for pid, (regex, severity) in PATTERNS.items():
        count = len(re.findall(regex, source))
        row[pid]                = count
        row[f'{pid}_detected']  = int(count > 0)
        risk_score             += count * SEVERITY_WEIGHT[severity]
        if severity == 'CRITICAL' and count > 0:
            any_critical = True

    row['Q_risk_score']       = risk_score
    row['Q_critical_count']   = sum(
        row[f'{pid}_detected']
        for pid,(_, sev) in PATTERNS.items() if sev=='CRITICAL')
    row['Q_high_count']       = sum(
        row[f'{pid}_detected']
        for pid,(_, sev) in PATTERNS.items() if sev=='HIGH')
    row['Q_medium_count']     = sum(
        row[f'{pid}_detected']
        for pid,(_, sev) in PATTERNS.items() if sev=='MEDIUM')
    row['quantum_vulnerable'] = int(risk_score > 0)
    row['quantum_critical']   = int(any_critical)

    results.append(row)

# ── Save ─────────────────────────────────────────────────
df = pd.DataFrame(results)
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/quantum_labels.csv', index=False)

# ── Summary ──────────────────────────────────────────────
total = len(df)
vuln  = df['quantum_vulnerable'].sum()
crit  = df['quantum_critical'].sum()
safe  = total - vuln

print("\n" + "="*55)
print("QUANTUM LABELING RESULTS")
print("="*55)
print(f"Total contracts scanned : {total:,}")
print(f"Quantum VULNERABLE      : {vuln:,}  ({vuln/total*100:.1f}%)")
print(f"Quantum CRITICAL only   : {crit:,}  ({crit/total*100:.1f}%)")
print(f"Quantum SAFE            : {safe:,}  ({safe/total*100:.1f}%)")

print("\nPattern breakdown:")
for pid in PATTERNS:
    n = df[f'{pid}_detected'].sum()
    print(f"  {pid:25}: {n:6,}  ({n/total*100:.1f}%)")

print("\nBy category:")
summary = (df.groupby('category')['quantum_vulnerable']
             .agg(['sum','count'])
             .rename(columns={'sum':'vuln','count':'total'}))
summary['pct'] = (summary['vuln']/summary['total']*100).round(1)
print(summary.to_string())

print("\n✅ Saved: data/processed/quantum_labels.csv")
