import pandas as pd
import os

print("Loading datasets...")
df_main   = pd.read_csv('data/raw/BCCC-SCsVul-2024.csv')
df_labels = pd.read_csv('data/processed/quantum_labels.csv')

print(f"Main   : {df_main.shape}")
print(f"Labels : {df_labels.shape}")

# Both have 111,897 rows — merge by position
df_labels = df_labels.reset_index(drop=True)
df_main   = df_main.reset_index(drop=True)

# Add quantum columns to main dataset
quantum_cols = [
    'filename', 'category',
    'Q1_ecrecover', 'Q2_oz_ecdsa', 'Q3_sig_checker',
    'Q4_eth_signed_hash', 'Q5_timestamp_prng', 'Q6_blockhash_prng',
    'Q7_sha256_auth', 'Q8_secp256k1', 'Q9_pubkey_storage',
    'Q1_ecrecover_detected', 'Q2_oz_ecdsa_detected',
    'Q3_sig_checker_detected', 'Q4_eth_signed_hash_detected',
    'Q5_timestamp_prng_detected', 'Q6_blockhash_prng_detected',
    'Q7_sha256_auth_detected', 'Q8_secp256k1_detected',
    'Q9_pubkey_storage_detected',
    'Q_risk_score', 'Q_critical_count',
    'Q_high_count', 'Q_medium_count',
    'quantum_vulnerable', 'quantum_critical'
]

df_merged = pd.concat([df_main, df_labels[quantum_cols]], axis=1)

print(f"\nMerged shape: {df_merged.shape}")

# ── Drop columns we don't need for ML ───────────────────
drop_cols = ['Unnamed: 0', 'ID',
             'Contract Information_0', 'Contract Information_1',
             'filename', 'category']
df_merged.drop(columns=drop_cols, inplace=True, errors='ignore')

# ── Handle missing values ────────────────────────────────
missing_before = df_merged.isnull().sum().sum()
df_merged.fillna(0, inplace=True)
missing_after  = df_merged.isnull().sum().sum()
print(f"Missing values: {missing_before} → {missing_after}")

# ── Feature matrix and labels ────────────────────────────
label_cols = [
    'Class01:ExternalBug', 'Class02:GasException',
    'Class03:MishandledException', 'Class04:Timestamp',
    'Class05:TransactionOrderDependence', 'Class06:UnusedReturn',
    'Class07:WeakAccessMod', 'Class08:CallToUnknown',
    'Class09:DenialOfService', 'Class10:IntegerUO',
    'Class11:Reentrancy', 'Class12:NonVulnerable',
    'quantum_vulnerable', 'quantum_critical',
    'Q_risk_score', 'Q_critical_count',
    'Q_high_count', 'Q_medium_count'
]

# Quantum-specific feature cols we added
quantum_feature_cols = [
    'Q1_ecrecover', 'Q2_oz_ecdsa', 'Q3_sig_checker',
    'Q4_eth_signed_hash', 'Q5_timestamp_prng', 'Q6_blockhash_prng',
    'Q7_sha256_auth', 'Q8_secp256k1', 'Q9_pubkey_storage',
    'Q1_ecrecover_detected', 'Q2_oz_ecdsa_detected',
    'Q3_sig_checker_detected', 'Q4_eth_signed_hash_detected',
    'Q5_timestamp_prng_detected', 'Q6_blockhash_prng_detected',
    'Q7_sha256_auth_detected', 'Q8_secp256k1_detected',
    'Q9_pubkey_storage_detected',
]

# Feature columns = everything except label cols
feature_cols = [c for c in df_merged.columns
                if c not in label_cols
                and c not in quantum_feature_cols]

print(f"\nFeature columns  : {len(feature_cols)}")
print(f"Quantum features : {len(quantum_feature_cols)}")
print(f"Total features   : {len(feature_cols)+len(quantum_feature_cols)}")

# ── Build X and y ────────────────────────────────────────
all_feature_cols = feature_cols + quantum_feature_cols
X = df_merged[all_feature_cols]
y = df_merged['quantum_vulnerable']

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"y distribution:")
print(f"  Vulnerable (1) : {y.sum():,}  ({y.mean()*100:.1f}%)")
print(f"  Safe       (0) : {(y==0).sum():,}  ({(1-y.mean())*100:.1f}%)")

# ── Save final datasets ──────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
X.to_csv('data/processed/X_features.csv', index=False)
y.to_csv('data/processed/y_labels.csv',   index=False)
df_merged.to_csv('data/processed/full_merged.csv', index=False)

print("\n✅ Saved:")
print("   data/processed/X_features.csv")
print("   data/processed/y_labels.csv")
print("   data/processed/full_merged.csv")
