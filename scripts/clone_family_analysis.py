"""
TDSC Upgrade: Clone-Family Propagation Analysis
Produces clone_family_stats.csv — quantifies how quantum
vulnerability propagates through contract clone families.

Uses ONLY pre-computed CSVs. No source file loading.
Safe to run on 8 GB RAM.
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("Clone-Family Quantum Vulnerability Propagation Analysis")
print("=" * 60)

# ── Load pre-computed data ────────────────────────────────
hashes = pd.read_csv('results/contract_hashes.csv')
labels = pd.read_csv('data/processed/quantum_labels.csv')

assert len(hashes) == len(labels), "Row count mismatch"

labels['content_hash'] = hashes['content_hash'].values
labels['quantum_broken'] = (
    (labels['Q1_ecrecover_detected'] == 1) |
    (labels['Q2_oz_ecdsa_detected']  == 1) |
    (labels['Q3_sig_checker_detected'] == 1)
).astype(int)

total = len(labels)
print(f"\nTotal contracts : {total:,}")
print(f"Quantum-broken  : {labels['quantum_broken'].sum():,}")

# ── Clone family enumeration ──────────────────────────────
# A "family" = set of contracts sharing an MD5 content hash
print("\n[1/5] Enumerating clone families...")
family_df = labels.groupby('content_hash').agg(
    family_size       = ('quantum_broken', 'count'),
    family_vulnerable = ('quantum_broken', 'sum'),
    category          = ('category', lambda x: x.mode()[0])
).reset_index()

# A family is "vulnerable" if ANY member is quantum-broken
family_df['is_vuln_family'] = (family_df['family_vulnerable'] > 0).astype(int)
family_df['is_clone_family'] = (family_df['family_size'] > 1).astype(int)

n_families        = len(family_df)
n_clone_families  = family_df['is_clone_family'].sum()
n_singleton       = n_families - n_clone_families
n_vuln_families   = family_df['is_vuln_family'].sum()
n_vuln_clone_fam  = (family_df['is_vuln_family'] & family_df['is_clone_family']).sum()

print(f"\n  Unique content-hash families : {n_families:,}")
print(f"  Singleton families (n=1)     : {n_singleton:,}")
print(f"  Clone families (n>1)         : {n_clone_families:,}")
print(f"  Vulnerable families          : {n_vuln_families:,}")
print(f"  Vulnerable clone families    : {n_vuln_clone_fam:,}")

# ── Amplification factor ──────────────────────────────────
print("\n[2/5] Computing vulnerability amplification...")
vuln_clone_mask = (family_df['is_vuln_family'] == 1) & (family_df['is_clone_family'] == 1)
vuln_clone_fam  = family_df[vuln_clone_mask]
total_cloned_vulnerable_instances = vuln_clone_fam['family_vulnerable'].sum()
total_unique_vuln_templates       = n_vuln_clone_fam  # one template per family

amplification_ratio = (
    total_cloned_vulnerable_instances / max(total_unique_vuln_templates, 1)
)
print(f"  Vulnerable instances in clone families : {total_cloned_vulnerable_instances:,}")
print(f"  Unique vulnerable templates            : {total_unique_vuln_templates:,}")
print(f"  Mean amplification ratio A_N/A_1       : {amplification_ratio:.2f}x")

# ── Top clone families ────────────────────────────────────
print("\n[3/5] Top 10 largest vulnerable clone families...")
top_fam = (vuln_clone_fam
           .sort_values('family_size', ascending=False)
           .head(10)[['content_hash','family_size','family_vulnerable','category']])
print(top_fam.to_string(index=False))

# ── Propagation rate ──────────────────────────────────────
print("\n[4/5] Clone propagation rates...")
# Among all contracts whose hash appears >1 time, what fraction are
# vulnerable because a sibling is vulnerable?
cloned_contracts = labels[labels['content_hash'].map(
    labels['content_hash'].value_counts()) > 1].copy()
cloned_contracts['family_has_vuln'] = cloned_contracts['content_hash'].map(
    family_df.set_index('content_hash')['is_vuln_family'])

propagated_vuln_contracts = cloned_contracts[
    (cloned_contracts['quantum_broken'] == 0) &
    (cloned_contracts['family_has_vuln'] == 1)
]
print(f"  Cloned contracts (non-unique) : {len(cloned_contracts):,}")
print(f"  Safe contracts in vuln family : {len(propagated_vuln_contracts):,}")
# NOTE: these are non-vulnerable siblings co-deployed alongside a vulnerable template

# ── Cluster size distribution ─────────────────────────────
print("\n[5/5] Clone family size distribution...")
size_bins = [1, 2, 5, 10, 50, 100, 500, 10000]
labels_b  = [1, '2-4', '5-9', '10-49', '50-99', '100-499', '500+']
family_df['size_bin'] = pd.cut(
    family_df['family_size'], bins=size_bins, labels=labels_b, right=False)
dist = family_df.groupby('size_bin', observed=True).agg(
    n_families=('family_size','count'),
    total_contracts=('family_size','sum'),
    vuln_families=('is_vuln_family','sum')
).reset_index()
print(dist.to_string(index=False))

# ── Save results ──────────────────────────────────────────
stats = {
    'total_contracts':                total,
    'unique_families':                n_families,
    'singleton_families':             int(n_singleton),
    'clone_families':                 int(n_clone_families),
    'vuln_families':                  int(n_vuln_families),
    'vuln_clone_families':            int(n_vuln_clone_fam),
    'cloned_vuln_instances':          int(total_cloned_vulnerable_instances),
    'unique_vuln_templates':          int(total_unique_vuln_templates),
    'mean_amplification_ratio':       round(amplification_ratio, 3),
}
pd.DataFrame(stats.items(), columns=['metric','value']).to_csv(
    'results/clone_family_stats.csv', index=False)
dist.to_csv('results/clone_size_distribution.csv', index=False)
print("\nSaved: results/clone_family_stats.csv")
print("Saved: results/clone_size_distribution.csv")
print("\n" + "="*60)
print("KEY FINDING: Every vulnerable template is deployed ≈",
      f"{amplification_ratio:.1f}x on average across the ecosystem.")
print("="*60)
