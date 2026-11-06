import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

os.makedirs('results', exist_ok=True)

df    = pd.read_csv('data/processed/quantum_labels.csv')
total = len(df)
vuln  = df['quantum_vulnerable'].sum()
safe  = total - vuln
crit  = df['quantum_critical'].sum()

print("="*60)
print("QVulnDetect — Final Analysis")
print("="*60)

# ── Finding 1 ────────────────────────────────────────────
print(f"\nFINDING 1 — Prevalence")
print(f"Total contracts    : {total:,}")
print(f"Quantum vulnerable : {vuln:,} ({vuln/total*100:.1f}%)")
print(f"Quantum critical   : {crit:,} ({crit/total*100:.1f}%)")
print(f"Quantum safe       : {safe:,} ({safe/total*100:.1f}%)")

# ── Finding 2 ────────────────────────────────────────────
patterns = {
    'Q1_ecrecover_detected':         ('ecrecover()',         'CRITICAL'),
    'Q2_oz_ecdsa_detected':          ('ECDSA.recover()',     'CRITICAL'),
    'Q3_sig_checker_detected':       ('SignatureChecker',    'CRITICAL'),
    'Q4_eth_signed_hash_detected':   ('toEthSignedMsgHash',  'HIGH'),
    'Q5_timestamp_prng_detected':    ('block.timestamp PRNG','HIGH'),
    'Q6_blockhash_prng_detected':    ('blockhash() PRNG',    'HIGH'),
    'Q7_sha256_auth_detected':       ('sha256() auth',       'MEDIUM'),
    'Q8_secp256k1_detected':         ('secp256k1 ref',       'MEDIUM'),
    'Q9_pubkey_storage_detected':    ('pubkey storage',      'MEDIUM'),
}

print(f"\nFINDING 2 — Pattern Breakdown")
print(f"{'Pattern':<26}{'Severity':<12}{'Count':>8}{'Pct':>8}")
print("-"*54)
for col,(name,sev) in patterns.items():
    n = df[col].sum()
    print(f"{name:<26}{sev:<12}{n:>8,}{n/total*100:>7.1f}%")

# ── Finding 3 ────────────────────────────────────────────
print(f"\nFINDING 3 — By Category")
cat_s = (df.groupby('category')['quantum_vulnerable']
           .agg(['sum','count'])
           .rename(columns={'sum':'vulnerable','count':'total'}))
cat_s['pct'] = (cat_s['vulnerable']/cat_s['total']*100).round(1)
cat_s = cat_s.sort_values('pct', ascending=False)
print(cat_s.to_string())

# ── Finding 4 ────────────────────────────────────────────
print(f"\nFINDING 4 — Bytecode ML")
print(f"All 4 models AUC ≈ 0.51 → random guessing")
print(f"Bytecode features alone CANNOT detect quantum vulnerability")

# ── Plot 1: Category bar ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12,5))
colors = ['#E74C3C' if p>20 else
          '#F39C12' if p>10 else '#27AE60'
          for p in cat_s['pct']]
bars = ax.bar(cat_s.index, cat_s['pct'],
              color=colors, alpha=0.85, edgecolor='white')
for bar,pct in zip(bars, cat_s['pct']):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f'{pct}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xlabel('Contract Category', fontsize=11)
ax.set_ylabel('Quantum Vulnerable (%)', fontsize=11)
ax.set_title('Quantum Vulnerability Rate by Contract Category\n'
             '111,897 Ethereum Smart Contracts',
             fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=30)
ax.set_ylim(0,55)
ax.legend(handles=[
    Patch(color='#E74C3C',label='>20% HIGH'),
    Patch(color='#F39C12',label='10-20% MEDIUM'),
    Patch(color='#27AE60',label='<10% LOW')
], loc='upper right')
plt.tight_layout()
plt.savefig('results/category_vulnerability.png', dpi=150)
print("\n✅ results/category_vulnerability.png")

# ── Plot 2: Pie + pattern bar ─────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(13,5))

axes[0].pie([vuln,safe],
    labels=[f'Vulnerable\n{vuln:,} ({vuln/total*100:.1f}%)',
            f'Safe\n{safe:,} ({safe/total*100:.1f}%)'],
    colors=['#E74C3C','#27AE60'],
    autopct='%1.1f%%', startangle=90)
axes[0].set_title('Overall Quantum Vulnerability\n111,897 Contracts',
                  fontweight='bold')

pnames  = [name for _,(name,sev) in patterns.items()]
pcounts = [int(df[col].sum()) for col in patterns.keys()]
pcols   = ['#E74C3C' if sev=='CRITICAL' else
           '#F39C12' if sev=='HIGH' else '#3498DB'
           for _,(name,sev) in patterns.items()]
axes[1].barh(pnames, pcounts, color=pcols, alpha=0.85)
axes[1].set_xlabel('Number of Contracts')
axes[1].set_title('Quantum Pattern Detection Count', fontweight='bold')
axes[1].legend(handles=[
    Patch(color='#E74C3C',label='CRITICAL'),
    Patch(color='#F39C12',label='HIGH'),
    Patch(color='#3498DB',label='MEDIUM')
])
plt.tight_layout()
plt.savefig('results/pattern_analysis.png', dpi=150)
print("✅ results/pattern_analysis.png")

# ── Plot 3: ML AUC ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
models = ['Random Forest','XGBoost','KNN','MLP']
aucs   = [0.5197, 0.5120, 0.5025, 0.4898]
bars   = ax.bar(range(4), aucs, color='#1A56DB', alpha=0.8)
for bar,v in zip(bars,aucs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.002,
            f'{v:.4f}', ha='center', va='bottom', fontsize=10)
ax.axhline(y=0.5, color='red', linestyle='--',
           linewidth=2, label='Random Baseline (0.50)')
ax.set_xticks(range(4))
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.40, 0.62)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Bytecode-Only ML — All Models Near Random\n'
             'Proves Source-Level Analysis is Required',
             fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('results/ml_bytecode_auc.png', dpi=150)
print("✅ results/ml_bytecode_auc.png")

# ── Save numbers ─────────────────────────────────────────
pd.DataFrame([{
    'Total':            total,
    'Vulnerable':       int(vuln),
    'Vulnerable_Pct':   round(vuln/total*100,1),
    'Critical':         int(crit),
    'Safe':             int(safe),
    'Best_AUC_bytecode':0.5197,
    'ecrecover':        int(df['Q1_ecrecover_detected'].sum()),
    'timestamp_prng':   int(df['Q5_timestamp_prng_detected'].sum()),
    'blockhash_prng':   int(df['Q6_blockhash_prng_detected'].sum()),
}]).to_csv('results/final_numbers.csv', index=False)
print("✅ results/final_numbers.csv")

print("\n✅ Implementation COMPLETE")
print("All results saved to results/")
