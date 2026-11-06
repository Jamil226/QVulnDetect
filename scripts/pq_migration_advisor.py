"""
TDSC Upgrade: PQ Migration Advisor
Rule-based post-quantum migration recommendation for each
detected Shor-vulnerable contract. Outputs a migration report.

Uses only quantum_labels.csv — no source loading, no ML.
Safe on any hardware.
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("Post-Quantum Migration Advisor")
print("NIST FIPS 204/205/206 recommendations per contract")
print("=" * 60)

labels = pd.read_csv('data/processed/quantum_labels.csv')
labels['quantum_broken'] = (
    (labels['Q1_ecrecover_detected'] == 1) |
    (labels['Q2_oz_ecdsa_detected']  == 1) |
    (labels['Q3_sig_checker_detected'] == 1)
).astype(int)

# Migration rules:
# Tier 1 (Q1, Q2, Q3): ECDSA → Lattice sig (CRYSTALS-Dilithium)
#   - For authentication: FIPS 204 (ML-DSA / Dilithium)
#   - For multi-sig / ERC-2612 permit patterns: Dilithium-based
#   - Note: EVM doesn't natively support PQ sigs yet; requires
#     precompile upgrade (EIP-7701 / account abstraction)
# Tier 2 (Q5 block.timestamp, Q6 blockhash): → commit-reveal PRNG
# Tier 3 (Q8 secp256k1 literal, Q9 pubkey): → audit surface only

def get_risk_level(row):
    if row['Q1_ecrecover_detected'] or row['Q2_oz_ecdsa_detected'] or \
       row['Q3_sig_checker_detected']:
        return 'CRITICAL'
    elif row['Q5_timestamp_prng_detected'] or row['Q6_blockhash_prng_detected']:
        return 'HIGH'
    elif row['Q8_secp256k1_detected'] or row['Q9_pubkey_storage_detected']:
        return 'MEDIUM'
    else:
        return 'NONE'

def get_migration_action(row):
    actions = []
    if row['Q1_ecrecover_detected']:
        actions.append('Replace ecrecover() with EIP-7701 PQ precompile '
                       'or account-abstraction validator (ML-DSA/FIPS-204)')
    if row['Q2_oz_ecdsa_detected']:
        actions.append('Replace OZ ECDSA.recover() with ML-DSA library '
                       'once available; pin OZ >= 5.x in interim')
    if row['Q3_sig_checker_detected']:
        actions.append('Replace SignatureChecker with PQ-compatible '
                       'signature validator module')
    if row['Q5_timestamp_prng_detected']:
        actions.append('Replace block.timestamp PRNG with commit-reveal '
                       'or Chainlink VRF (classical hardening)')
    if row['Q6_blockhash_prng_detected']:
        actions.append('Replace blockhash PRNG with VRF oracle '
                       '(classical hardening)')
    if row['Q8_secp256k1_detected']:
        actions.append('Audit secp256k1 literal usage; document PQ migration plan')
    if row['Q9_pubkey_storage_detected']:
        actions.append('Flag on-chain public key storage for HNDL audit')
    return ' | '.join(actions) if actions else 'No action required'

def get_pq_standard(row):
    if row['Q1_ecrecover_detected'] or row['Q2_oz_ecdsa_detected'] or \
       row['Q3_sig_checker_detected']:
        return 'NIST FIPS 204 (ML-DSA/Dilithium) + EIP-7701'
    elif row['Q5_timestamp_prng_detected'] or row['Q6_blockhash_prng_detected']:
        return 'Classical fix (VRF) — Grover mitigation is low priority'
    else:
        return '—'

print("\n[1/4] Computing risk levels...")
labels['risk_level']       = labels.apply(get_risk_level, axis=1)
labels['migration_action'] = labels.apply(get_migration_action, axis=1)
labels['pq_standard']      = labels.apply(get_pq_standard, axis=1)

print("\n[2/4] Risk level distribution:")
risk_dist = labels['risk_level'].value_counts()
print(risk_dist.to_string())

print("\n[3/4] Migration complexity summary:")
critical = labels[labels['risk_level'] == 'CRITICAL']
high     = labels[labels['risk_level'] == 'HIGH']
medium   = labels[labels['risk_level'] == 'MEDIUM']

print(f"\n  CRITICAL (direct FIPS 204 migration)  : {len(critical):>6,}")
print(f"  HIGH     (classical VRF hardening)     : {len(high):>6,}")
print(f"  MEDIUM   (audit surface documentation) : {len(medium):>6,}")

# Category breakdown for CRITICAL
print("\n[4/4] CRITICAL contracts by category:")
cat_crit = critical.groupby('category').size().sort_values(ascending=False)
print(cat_crit.to_string())

# Save full migration report (critical contracts only)
migration_report = critical[['filename','category',
                              'risk_level','pq_standard','migration_action',
                              'Q_risk_score']].copy()
migration_report = migration_report.sort_values('Q_risk_score', ascending=False)
migration_report.to_csv('results/pq_migration_report.csv', index=False)

# Summary table for paper
summary = pd.DataFrame([
    {'Risk_Level': 'CRITICAL',
     'n_contracts': len(critical),
     'NIST_Standard': 'FIPS 204 (ML-DSA)',
     'EVM_Path': 'EIP-7701 / Account Abstraction',
     'HNDL_Urgency': 'Immediate'},
    {'Risk_Level': 'HIGH',
     'n_contracts': len(high),
     'NIST_Standard': 'VRF / Classical fix',
     'EVM_Path': 'Chainlink VRF / commit-reveal',
     'HNDL_Urgency': 'Medium-term'},
    {'Risk_Level': 'MEDIUM',
     'n_contracts': len(medium),
     'NIST_Standard': 'Audit documentation',
     'EVM_Path': 'Risk documentation',
     'HNDL_Urgency': 'Low'},
])
summary.to_csv('results/pq_migration_summary.csv', index=False)

print("\nSaved: results/pq_migration_report.csv")
print("Saved: results/pq_migration_summary.csv")
print(f"\nTotal contracts requiring PQ migration action: "
      f"{len(critical)+len(high):,} "
      f"({(len(critical)+len(high))/len(labels)*100:.1f}%)")
