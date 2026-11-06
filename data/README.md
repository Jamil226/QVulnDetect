# QVulnDetect Dataset Architecture

This directory contains the dataset artifacts supporting the paper:
> **QVulnDetect: How Quantum-Vulnerable Cryptographic Dependencies Propagate Through Ethereum Smart Contract Cloning—Taxonomy, Ecosystem-Scale Measurement, and Adversarial Detection Analysis**

---

## 1. Directory Overview

```
data/
├── processed/
│   ├── X_features.csv        # Pre-extracted 236 numerical opcode & structural features (111,897 contracts)
│   ├── y_labels.csv          # Binary quantum-vulnerability target labels (1 = Shor/ECDSA vulnerable)
│   ├── quantum_labels.csv    # Granular QVPT detection flags across cryptographic dependency types
│   └── full_merged.csv       # Merged feature matrix with quantum labels for end-to-end evaluation
└── raw/
    ├── BCCC-SCsVul-2024.csv  # Base feature dataset from York University BCCC Research Centre
    └── Source Codes/         # (Optional) Raw Solidity source codes for all 111,897 smart contracts
```

---

## 2. Immediate Replication (No Large Downloads Required)

To ensure rapid reproducibility for journal reviewers and researchers without requiring gigabyte-scale source code downloads, all model training, cross-family generalization, and evaluation pipelines can be run directly using the pre-extracted matrices in `data/processed/`:

- **`X_features.csv`**: Contains 236 numeric features (138 EVM opcode frequency counts, bytecode entropy metrics, ABI metrics, and AST structural attributes). All classical vulnerability leak features and label proxy columns were strictly excluded to ensure orthogonal quantum vulnerability evaluation.
- **`y_labels.csv`**: Binary classification labels verified against a 500-contract stratified validation sample with Cohen's $\kappa = 0.984$.
- **`quantum_labels.csv`**: Flags indicating exposure to Tier-1 Shor-vulnerable ECDSA (`ecrecover`, OpenZeppelin ECDSA libraries, signature verification templates).

Running:
```bash
python scripts/train_models.py
python scripts/clone_family_analysis.py
python scripts/pq_migration_advisor.py
```
operates entirely on `data/processed/` and `results/`.

---

## 3. Obtaining the Raw Solidity Source Codes (Optional)

The raw Solidity source codes (`Source Codes/`, 2.2 GB uncompressed, 384 MB compressed) originate from the **BCCC-SCsVul-2024** benchmark curated by York University's Blockchain Connected and Cognitive Technologies (BCCC) Research Centre.

Due to GitHub's file size constraints (100 MB per-file limit), the raw `.sol` archive is not stored directly inside this Git repository.

### How to Download & Set Up Raw Contracts:
1. Access the official dataset repository at York University:
   - **URL**: [https://www.yorku.ca/research/bccc/](https://www.yorku.ca/research/bccc/)
2. Download `SourceCodes.zip` (or the BCCC-SCsVul-2024 source codes package).
3. Unzip the archive into:
   ```bash
   data/raw/Source Codes/
   ```
4. Once unzipped, source-level NLP pipelines (`scripts/tfidf_model.py`, `scripts/adversarial_test.py`, `scripts/minhash_dedup.py`) can be executed.

---

## 4. Citation

If using this dataset structure, please cite:

```bibtex
@misc{bccc2024,
  author = {{BCCC Research Centre}},
  title  = {{BCCC-SCsVul-2024}: Smart Contracts Vulnerabilities Dataset},
  year   = {2024},
  note   = {York University, Canada. 111,897 real Ethereum smart contracts. Available: https://www.yorku.ca/research/bccc/}
}
```
