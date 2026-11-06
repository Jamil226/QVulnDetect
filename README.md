# QVulnDetect: How Quantum-Vulnerable Cryptographic Dependencies Propagate Through Ethereum Smart Contract Cloning

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/Dataset-BCCC--SCsVul--2024-green.svg)](https://www.yorku.ca/research/bccc/)
[![EVM Standards](https://img.shields.io/badge/Standards-NIST%20FIPS%20204%2F205%20%7C%20EIP--7701-orange.svg)](https://csrc.nist.gov/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](#replication-guide)

---

## Abstract & Overview

The Elliptic Curve Digital Signature Algorithm (**ECDSA**) over the `secp256k1` curve underlies identity, authentication, and asset transfer across Ethereum smart contracts. Shor's polynomial-time quantum algorithm breaks this primitive, exposing cryptographic dependencies to Harvest Now, Decrypt Later (**HNDL**) threats. Yet, the ecosystem-scale exposure and the degree to which code duplication amplifies this risk have remained unmeasured.

**QVulnDetect** presents the first ecosystem-scale measurement and propagation study of quantum-vulnerable cryptographic dependencies across **111,897 real Ethereum smart contracts**. Our study formalizes and validates three core propositions:

1. **Linear Vulnerability Scaling:** Vulnerability exposure scales linearly with contract clone-family size.
2. **Ecosystem Amplification:** Per-template exposure amplifies by a mean factor of **2.62$\times$** across 66,312 content-hash clone families.
3. **Adversarial Structural Evasion:** Source-level lexical and NLP models suffer a catastrophic drop in recall (falling to 0%) under adversarial evasion (e.g., inline assembly, proxy delegation, variable obfuscation), whereas EVM bytecode precompile tracing provides evasion-resistant detection.

Furthermore, evaluation on near-duplicate-free test splits (Jaccard similarity $\geq$ 0.80, $n = 8{,}814$) demonstrates an **AUC of 0.9972**, confirming that model performance is genuine and not an artifact of clone contamination. Finally, we provide an automated rule-based Post-Quantum Migration Advisor mapping detected vulnerabilities to **NIST FIPS 204 (ML-DSA)**, **NIST FIPS 205 (SLH-DSA)**, and Account Abstraction (**EIP-7701**).

---

## Repository Structure

```
QVulnDetect/
├── data/
│   ├── README.md                  # Comprehensive dataset & feature schema guide
│   ├── processed/
│   │   ├── X_features.csv         # 236 numerical opcode & structural features (111,897 contracts)
│   │   ├── y_labels.csv           # Binary quantum vulnerability labels (1 = Shor/ECDSA vulnerable)
│   │   ├── quantum_labels.csv     # Granular QVPT detection flags across dependency categories
│   │   └── full_merged.csv        # Merged dataset linking feature matrix with labels
│   └── raw/
│       └── BCCC-SCsVul-2024.csv   # Base vulnerability dataset from York University BCCC
├── models/                        # Serialized model checkpoints directory
├── results/                       # Empirical evaluation outputs, CSV tables, and paper figures
│   ├── adversarial_robustness.csv # Evasion robustness under structural perturbations
│   ├── auc_confidence_intervals.csv # Hanley-McNeil AUC confidence intervals
│   ├── clone_analysis.csv         # Exact duplicate and template replication metrics
│   ├── clone_family_stats.csv     # Clone-family vulnerability propagation distributions
│   ├── dedup_evaluation.csv       # Disjoint train/test deduplication evaluation results
│   ├── deep_masking.csv           # Deep token masking sensitivity analysis
│   ├── feature_importance.csv     # LR / Tree feature attribution scores
│   ├── final_numbers.csv          # Ecosystem-scale headline statistics
│   ├── minhash_dedup.csv          # MinHash-LSH near-duplicate cluster statistics
│   ├── model_comparison.csv       # Multi-classifier performance comparison
│   ├── near_dedup_auc.csv         # Dedup-resistant ROC-AUC comparison table
│   ├── ood_generalization.csv     # Leave-One-Category-Out cross-domain evaluation
│   ├── pq_migration_report.csv    # NIST FIPS 204/205 per-contract migration roadmap
│   ├── pq_migration_summary.csv   # Aggregate ecosystem migration breakdown
│   ├── shap_summary.csv           # SHAP cryptographic feature attributions
│   └── *.png                      # Publication-grade figures (ROC curves, confusion matrices, etc.)
├── scripts/                       # End-to-end experimental and evaluation scripts
│   ├── adversarial_test.py        # Evaluates model recall under adversarial code transformations
│   ├── auc_confidence.py          # Computes Hanley-McNeil parametric confidence intervals
│   ├── bootstrap_ci.py            # Computes non-parametric bootstrap confidence intervals (B=1000)
│   ├── bytecode_analyzer.py       # EVM opcode disassembler and STATICCALL scanner
│   ├── bytecode_scanner_prototype.py # Direct bytecode precompile 0x01 heuristic scanner
│   ├── clone_analysis.py          # Exact MD5 hash family clustering
│   ├── clone_family_analysis.py   # Amplification factor and propagation scaling analysis
│   ├── cross_family_generalization.py # Cross-family template generalization experiments
│   ├── dedup_evaluation.py        # Strict repository-disjoint deduplication benchmark
│   ├── deep_masking.py            # Token and identifier masking robustness
│   ├── explore.py                 # Exploratory data analysis of contract distributions
│   ├── feature_importance.py      # Opcode and structural feature coefficient rankings
│   ├── final_analysis.py          # Synthesis of ecosystem-scale vulnerability statistics
│   ├── merge_dataset.py           # Merges BCCC features with QVPT ground-truth labels
│   ├── minhash_dedup.py           # MinHash LSH near-clone detection (Jaccard >= 0.80)
│   ├── near_dedup_auc.py          # Near-duplicate resistant AUC generation
│   ├── ood_generalization.py      # Out-of-distribution category evaluation
│   ├── pq_migration_advisor.py    # Automated NIST FIPS 204/205 & EIP-7701 advisor
│   ├── precompile_baseline.py     # Deterministic 0x01 ecrecover precompile baseline
│   ├── quantum_labeler.py         # QVPT ground-truth labeling pipeline
│   ├── regex_baseline.py          # Lexical regex matching baseline
│   ├── rpc_scanner.py             # Live JSON-RPC Ethereum node bytecode tracer
│   ├── shap_analysis.py           # SHAP TreeExplainer feature importance
│   ├── tfidf_final.py             # Source-level NLP TF-IDF classifier
│   ├── token_masking.py           # Token masking sensitivity pipeline
│   ├── train_experiment_b.py      # Source-level AST feature evaluation
│   ├── train_models.py            # Primary ML classification benchmark (RF, XGB, KNN, MLP)
│   ├── truncation_sensitivity.py  # Input sequence truncation impact analysis
│   ├── update_paper.py            # LaTeX manuscript figures and tables synchronization
│   ├── validate_labels_500.py     # Stratified 500-contract validation (Cohen's kappa = 0.984)
│   └── validate_labels.py         # 100-contract initial validation sampler
├── LICENSE                        # MIT License
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Project documentation and reproduction guide
```

---

## Key Empirical Findings

| Metric                                         |            Full Test Set ($N=22{,}380$)            | Deduplicated Test Set ($N=9{,}261$) |
| :--------------------------------------------- | :------------------------------------------------: | :---------------------------------: |
| **Random Forest (AUC)**                        |                     **0.9968**                     |             **0.9975**              |
| **Random Forest (F1-Score)**                   |                       0.7420                       |               0.7680                |
| **Random Forest (Precision)**                  |                       0.8810                       |               0.8950                |
| **Random Forest (Recall)**                     |                       0.6410                       |               0.6720                |
| **XGBoost (AUC)**                              |                     **0.9965**                     |             **0.9972**              |
| **Near-Dedup Resistant AUC ($Jac \geq 0.80$)** |                         —                          |             **0.9972**              |
| **Clone Amplification Factor**                 | **2.62$\times$** mean across 66,312 clone families |                  —                  |
| **Cohen's $\kappa$ Labelling Agreement**       |     **0.984** ($n = 500$, stratified seed 99)      |                  —                  |

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Linux / macOS / Windows with Bash

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/ziezeeshan/QVulnDetect.git
cd QVulnDetect

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Replication Guide

All experiments reported in the paper can be replicated directly using the provided scripts and pre-extracted features:

### 1. Model Training & Baseline Evaluation (Experiment A)

Trains Random Forest, XGBoost, KNN, Multi-Layer Perceptron, and Logistic Regression on the 236 orthogonal opcode features:

```bash
python scripts/train_models.py
```

_Outputs: Accuracy, Precision, Recall, F1, and ROC-AUC comparisons saved to `results/model_comparison.csv` and ROC plot to `results/ml_bytecode_auc.png`._

### 2. Clone-Family & Vulnerability Propagation Analysis

Quantifies how quantum-vulnerable templates replicate across the Ethereum ecosystem:

```bash
python scripts/clone_family_analysis.py
```

_Outputs: Amplification statistics, family size distributions, and propagation metrics to `results/clone_family_stats.csv`._

### 3. Deduplication & Near-Clone Evaluation

Evaluates whether clone contamination inflates detection performance:

```bash
python scripts/near_dedup_auc.py
```

_Outputs: Near-duplicate-resistant AUC table matching the paper in `results/near_dedup_auc.csv`._

### 4. Statistical Rigor: Hanley-McNeil & Bootstrap Confidence Intervals

Generates 95% confidence intervals for all reported AUC values:

```bash
python scripts/auc_confidence.py
```

_Outputs: Parametric Hanley-McNeil confidence intervals in `results/auc_confidence_intervals.csv`._

### 5. Deterministic Bytecode Precompile Baseline

Evaluates the EVM precompile address `0x01` (`ecrecover`) call scanner:

```bash
python scripts/precompile_baseline.py
```

_Outputs: Detection precision, recall, and runtime efficiency in `results/precompile_baseline.csv`._

### 6. Post-Quantum Migration Advisor

Applies the rule-based decision framework mapping each detected contract to NIST FIPS 204 (ML-DSA) and EVM migration pathways:

```bash
python scripts/pq_migration_advisor.py
```

_Outputs: Comprehensive contract-level migration audit to `results/pq_migration_report.csv` and summary breakdown to `results/pq_migration_summary.csv`._

### 7. Feature Importance & Interpretability

Extracts feature attribution rankings and SHAP importance:

```bash
python scripts/feature_importance.py
```

_Outputs: Top opcode importance rankings and plots in `results/feature_importance.csv` and `results/feature_importance.png`._

---

## Dataset & Feature Engineering

- **Source:** York University BCCC Research Centre (`BCCC-SCsVul-2024`), comprising 111,897 verified Ethereum smart contracts.
- **Orthogonal Features:** 236 numerical attributes extracted from EVM bytecode, including 138 opcode frequencies, instruction entropy, ABI interfaces, and AST metrics.
- **Leakage Prevention:** All classical vulnerability labels (e.g., Reentrancy, Overflow) and source-level proxy columns were strictly excluded to ensure the model learns cryptographic architecture rather than superficial correlations.
- **Raw Source Code:** Full raw Solidity source files (2.2 GB) can be retrieved as documented in [data/README.md](data/README.md).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
