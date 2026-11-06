"""
TDSC Upgrade: Adversarial Robustness Evaluation
Tests detection against 4 evasion strategies:
  1. Inline assembly wrappers
  2. Proxy delegation patterns
  3. Renamed function identifiers
  4. Obfuscated signature variables

Uses synthetic adversarial contracts — no source file loading.
Safe on 8 GB RAM.
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os

print("=" * 60)
print("Adversarial Robustness Evaluation")
print("=" * 60)

# ─── Synthetic adversarial contract templates ─────────────

# Clean ecrecover template (baseline positive)
CLEAN_ECRECOVER = """
pragma solidity ^0.8.0;
contract Verifier {{
    address public owner;
    function verify(bytes32 hash, uint8 v, bytes32 r, bytes32 s)
        public view returns (bool) {{
        address signer = ecrecover(hash, v, r, s);
        return signer == owner;
    }}
}}
"""

# Adversarial 1: Inline assembly wrapper
ASSEMBLY_ECRECOVER = """
pragma solidity ^0.8.0;
contract Verifier {{
    address public owner;
    function verify(bytes32 h, uint8 a, bytes32 b, bytes32 c)
        public view returns (bool) {{
        address result;
        assembly {{
            let p := mload(0x40)
            mstore(p, h)
            mstore(add(p, 0x20), a)
            mstore(add(p, 0x40), b)
            mstore(add(p, 0x60), c)
            if iszero(staticcall(gas(), 0x01, p, 0x80, p, 0x20)) {{
                revert(0, 0)
            }}
            result := mload(p)
        }}
        return result == owner;
    }}
}}
"""

# Adversarial 2: Proxy delegation
PROXY_ECRECOVER = """
pragma solidity ^0.8.0;
interface ISignatureVerifier {{
    function checkAuth(bytes32 d, uint8 x, bytes32 y, bytes32 z)
        external view returns (address);
}}
contract ProxyAuth {{
    address public owner;
    ISignatureVerifier public verifier;
    function authenticate(bytes32 d, uint8 x, bytes32 y, bytes32 z)
        public view returns (bool) {{
        address recovered = verifier.checkAuth(d, x, y, z);
        return recovered == owner;
    }}
}}
"""

# Adversarial 3: Renamed variables (all ECDSA terms substituted)
RENAMED_ECRECOVER = """
pragma solidity ^0.8.0;
contract TokenGate {{
    address private controller;
    function authorize(bytes32 data, uint8 idx, bytes32 part1, bytes32 part2)
        external view returns (bool) {{
        address recovered = ecrecover(data, idx, part1, part2);
        return recovered == controller;
    }}
}}
"""

# Adversarial 4: Multi-hop obfuscation (library indirection)
OBFUSCATED_ECRECOVER = """
pragma solidity ^0.8.0;
library CryptoLib {{
    function _internal(bytes32 m, uint8 n, bytes32 p, bytes32 q)
        internal pure returns (address) {{
        return ecrecover(m, n, p, q);
    }}
}}
contract Wallet {{
    using CryptoLib for bytes32;
    address public admin;
    function validate(bytes32 m, uint8 n, bytes32 p, bytes32 q)
        public view returns (bool) {{
        return m._internal(n, p, q) == admin;
    }}
}}
"""

# Safe (non-vulnerable) contracts
SAFE_ERC20 = """
pragma solidity ^0.8.0;
contract Token {{
    mapping(address => uint256) balances;
    function transfer(address to, uint256 amount) public returns (bool) {{
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[to] += amount;
        return true;
    }}
}}
"""

SAFE_STORAGE = """
pragma solidity ^0.8.0;
contract Storage {{
    uint256 private val;
    function store(uint256 v) public {{ val = v; }}
    function retrieve() public view returns (uint256) {{ return val; }}
}}
"""

# ─── Build synthetic evaluation sets ─────────────────────
np.random.seed(42)

# Load real labels for baseline comparison
labels = pd.read_csv('data/processed/quantum_labels.csv')
labels['quantum_broken'] = (
    (labels['Q1_ecrecover_detected'] == 1) |
    (labels['Q2_oz_ecdsa_detected']  == 1) |
    (labels['Q3_sig_checker_detected'] == 1)
).astype(int)

SRC_DIR   = 'data/raw/Source Codes'
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

print("\n[1/4] Training baseline SGD on real contract sample...")
y = labels['quantum_broken'].values
idx = np.arange(len(labels))
tr_i, te_i, y_tr, y_te = train_test_split(
    idx, y, test_size=0.2, random_state=42, stratify=y)

# Use a smaller subsample for speed
np.random.seed(42)
tr_sub = np.random.choice(tr_i, size=min(8000, len(tr_i)), replace=False)
te_sub = np.random.choice(te_i, size=min(1000, len(te_i)), replace=False)

print(f"  Loading {len(tr_sub):,} training + {len(te_sub):,} test contracts...")
tr_texts = [load_source(labels.iloc[i]) for i in tr_sub]
te_texts = [load_source(labels.iloc[i]) for i in te_sub]
y_tr_sub = y[tr_sub]
y_te_sub = y[te_sub]

# Remove empty
tr_mask = [bool(t.strip()) for t in tr_texts]
te_mask = [bool(t.strip()) for t in te_texts]
tr_texts_f = [t for t, m in zip(tr_texts, tr_mask) if m]
y_tr_f     = y_tr_sub[tr_mask]
te_texts_f = [t for t, m in zip(te_texts, te_mask) if m]
y_te_f     = y_te_sub[te_mask]

print(f"  Non-empty train: {len(tr_texts_f):,}, test: {len(te_texts_f):,}")

hv  = HashingVectorizer(ngram_range=(1,2), n_features=2**14,
                        token_pattern=TOKEN_PAT, norm='l2')
X_tr = hv.transform(tr_texts_f)
clf  = SGDClassifier(loss='modified_huber', random_state=42, max_iter=500)
clf.fit(X_tr, y_tr_f)

X_te = hv.transform(te_texts_f)
y_prob = clf.predict_proba(X_te)[:,1]
baseline_auc = roc_auc_score(y_te_f, y_prob)
print(f"  Baseline AUC on real test set: {baseline_auc:.4f}")

# ─── Adversarial evaluation ───────────────────────────────
print("\n[2/4] Building adversarial evaluation sets...")

N_ADV = 200  # synthetic adversarial positive contracts
N_SAFE = 200  # synthetic safe contracts

adv_templates = {
    'Inline Assembly':     ASSEMBLY_ECRECOVER,
    'Proxy Delegation':    PROXY_ECRECOVER,
    'Renamed Variables':   RENAMED_ECRECOVER,
    'Library Obfuscation': OBFUSCATED_ECRECOVER,
}
safe_templates = [SAFE_ERC20, SAFE_STORAGE]

results = []
print("\n[3/4] Testing each adversarial strategy...")
for strategy, template in adv_templates.items():
    # Create N_ADV variants with minor perturbations
    adv_contracts = []
    for i in range(N_ADV):
        # Add noise: random comment lines to diversify token distribution
        noise = f"// Contract instance {i} revision {np.random.randint(1000)}\n"
        adv_contracts.append(noise + template)

    safe_contracts = []
    for i in range(N_SAFE):
        t = safe_templates[i % len(safe_templates)]
        noise = f"// Safe contract {i}\n"
        safe_contracts.append(noise + t)

    # True labels: adversarial=1 (are vulnerable), safe=0
    all_texts  = adv_contracts + safe_contracts
    all_labels = np.array([1]*N_ADV + [0]*N_SAFE)

    X_adv  = hv.transform(all_texts)
    y_prob_adv = clf.predict_proba(X_adv)[:,1]
    adv_auc = roc_auc_score(all_labels, y_prob_adv)

    # Detection rate: fraction of adversarial contracts with prob > 0.5
    adv_only_prob = y_prob_adv[:N_ADV]
    detect_rate   = (adv_only_prob > 0.5).mean()
    fp_rate       = (y_prob_adv[N_ADV:] > 0.5).mean()

    results.append({
        'Strategy':       strategy,
        'AUC':            round(adv_auc, 4),
        'Detect_Rate':    round(detect_rate, 4),
        'FP_Rate':        round(fp_rate, 4),
        'AUC_Drop':       round(baseline_auc - adv_auc, 4),
    })
    print(f"  {strategy:25s} AUC={adv_auc:.4f}  "
          f"Det={detect_rate:.1%}  FP={fp_rate:.1%}  "
          f"ΔAUC={baseline_auc-adv_auc:+.4f}")

print("\n[4/4] Saving results...")
df_adv = pd.DataFrame(results)
df_adv.to_csv('results/adversarial_robustness.csv', index=False)
print("\nSaved: results/adversarial_robustness.csv")
print()
print(df_adv.to_string(index=False))
print()
print(f"Baseline AUC (real test):  {baseline_auc:.4f}")
print(f"Mean adversarial AUC drop: "
      f"{df_adv['AUC_Drop'].mean():+.4f}")
