import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.utils import resample
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("QVulnDetect — Experiment B (Source-Level Features)")
print("="*60)

# ── Load quantum labels (source-level features) ──────────
print("\n[1/7] Loading quantum labels...")
df_labels = pd.read_csv('data/processed/quantum_labels.csv')
print(f"Quantum labels shape: {df_labels.shape}")

# ── Build feature matrix from SOURCE-LEVEL patterns ──────
# These are COUNTS from source code — legitimate features
# NOT the binary detected flags (those created the label)
print("\n[2/7] Building source-level feature matrix...")

source_features = [
    'Q1_ecrecover',        # count of ecrecover calls
    'Q2_oz_ecdsa',         # count of OZ ECDSA calls
    'Q3_sig_checker',      # count of SignatureChecker calls
    'Q4_eth_signed_hash',  # count of toEthSignedMessageHash
    'Q5_timestamp_prng',   # count of block.timestamp
    'Q6_blockhash_prng',   # count of blockhash()
    'Q7_sha256_auth',      # count of sha256()
    'Q8_secp256k1',        # count of secp256k1 references
    'Q9_pubkey_storage',   # count of pubkey storage patterns
    'Q_risk_score',        # weighted severity score
    'Q_critical_count',    # number of critical pattern types
    'Q_high_count',        # number of high pattern types
    'Q_medium_count',      # number of medium pattern types
]

# Also load bytecode features to combine
X_bytecode = pd.read_csv('data/processed/X_features.csv')
non_numeric = X_bytecode.select_dtypes(exclude=[np.number]).columns
X_bytecode  = X_bytecode.drop(columns=non_numeric).fillna(0)
X_bytecode.columns = [str(c).replace('[','_').replace(']','_')
                             .replace('<','_').replace('>','_')
                             .replace(' ','_') for c in X_bytecode.columns]
direct_cols = [c for c in X_bytecode.columns if c.startswith('Q')]
X_bytecode  = X_bytecode.drop(columns=direct_cols)

# Source features from quantum_labels.csv
X_source = df_labels[source_features].fillna(0)
X_source = X_source.reset_index(drop=True)
X_bytecode = X_bytecode.reset_index(drop=True)

# Combined: bytecode + source patterns
X_combined = pd.concat([X_bytecode, X_source], axis=1)

# Label
y = df_labels['quantum_vulnerable'].reset_index(drop=True)

print(f"Bytecode features  : {X_bytecode.shape[1]}")
print(f"Source features    : {X_source.shape[1]}")
print(f"Combined features  : {X_combined.shape[1]}")
print(f"Label distribution : {y.value_counts().to_dict()}")

# ── Three feature sets to compare ────────────────────────
experiments = {
    'A: Bytecode Only':    X_bytecode,
    'B: Source Only':      X_source,
    'C: Bytecode+Source':  X_combined,
}

exp_results = {}

for exp_name, X in experiments.items():
    print(f"\n{'='*50}")
    print(f"Running {exp_name}")
    print(f"{'='*50}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Balance
    df_tr  = pd.concat([X_train, y_train], axis=1)
    label_col = y_train.name
    df_maj = df_tr[df_tr[label_col]==0]
    df_min = df_tr[df_tr[label_col]==1]
    df_min_up = resample(df_min, replace=True,
                         n_samples=len(df_maj)//2, random_state=42)
    df_bal      = pd.concat([df_maj, df_min_up])
    X_train_bal = df_bal.drop(columns=[label_col])
    y_train_bal = df_bal[label_col]

    # Scale
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_bal)
    X_test_sc  = scaler.transform(X_test)
    X_train_np = X_train_bal.values
    X_test_np  = X_test.values

    # Best model per experiment: Random Forest + XGBoost
    models = {
        'Random Forest': (
            RandomForestClassifier(n_estimators=200, max_depth=15,
                class_weight='balanced', random_state=42, n_jobs=-1),
            'df'),
        'XGBoost': (
            XGBClassifier(n_estimators=200, max_depth=6,
                learning_rate=0.05, subsample=0.8,
                scale_pos_weight=6, eval_metric='logloss',
                random_state=42, verbosity=0),
            'np'),
        'KNN': (
            KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            'sc'),
        'MLP': (
            MLPClassifier(hidden_layer_sizes=(128,64),
                max_iter=100, random_state=42,
                early_stopping=True, verbose=False),
            'sc'),
    }

    exp_results[exp_name] = {}
    for mname, (model, dtype) in models.items():
        t0 = time.time()
        if dtype == 'sc':
            model.fit(X_train_sc, y_train_bal)
            yp  = model.predict(X_test_sc)
            ypr = model.predict_proba(X_test_sc)[:,1]
        elif dtype == 'np':
            model.fit(X_train_np, y_train_bal.values)
            yp  = model.predict(X_test_np)
            ypr = model.predict_proba(X_test_np)[:,1]
        else:
            model.fit(X_train_bal, y_train_bal)
            yp  = model.predict(X_test)
            ypr = model.predict_proba(X_test)[:,1]

        elapsed = time.time()-t0
        r = {
            'acc':  accuracy_score(y_test, yp),
            'f1':   f1_score(y_test, yp, zero_division=0),
            'prec': precision_score(y_test, yp, zero_division=0),
            'rec':  recall_score(y_test, yp, zero_division=0),
            'auc':  roc_auc_score(y_test, ypr),
            'cm':   confusion_matrix(y_test, yp),
            'time': elapsed,
        }
        exp_results[exp_name][mname] = r
        print(f"  {mname:<15} Acc:{r['acc']:.4f}  "
              f"F1:{r['f1']:.4f}  AUC:{r['auc']:.4f}  "
              f"({elapsed:.1f}s)")

# ── Master Summary ────────────────────────────────────────
print("\n" + "="*60)
print("MASTER COMPARISON — All Experiments vs All Models")
print("="*60)
print(f"{'Experiment':<25}{'Model':<18}"
      f"{'Acc':>7}{'F1':>7}{'Prec':>7}{'Rec':>7}{'AUC':>7}")
print("-"*75)
for exp, models_res in exp_results.items():
    for mname, r in models_res.items():
        print(f"{exp:<25}{mname:<18}"
              f"{r['acc']:>7.4f}{r['f1']:>7.4f}"
              f"{r['prec']:>7.4f}{r['rec']:>7.4f}"
              f"{r['auc']:>7.4f}")
    print()

# ── Best overall ─────────────────────────────────────────
best_auc  = 0
best_info = {}
for exp, models_res in exp_results.items():
    for mname, r in models_res.items():
        if r['auc'] > best_auc:
            best_auc  = r['auc']
            best_info = {'exp':exp,'model':mname,'r':r}

print(f"🏆 BEST OVERALL")
print(f"   Experiment : {best_info['exp']}")
print(f"   Model      : {best_info['model']}")
print(f"   F1         : {best_info['r']['f1']:.4f}")
print(f"   AUC-ROC    : {best_info['r']['auc']:.4f}")

# ── Save results ─────────────────────────────────────────
rows = []
for exp, models_res in exp_results.items():
    for mname, r in models_res.items():
        rows.append({'Experiment':exp,'Model':mname,
                     'Accuracy':r['acc'],'F1':r['f1'],
                     'Precision':r['prec'],'Recall':r['rec'],
                     'AUC':r['auc']})
pd.DataFrame(rows).to_csv('results/experiment_comparison.csv',
                           index=False)

# ── Plot: AUC comparison across experiments ──────────────
fig, ax = plt.subplots(figsize=(12,6))
exp_names  = list(exp_results.keys())
model_names = ['Random Forest','XGBoost','KNN','MLP']
x      = np.arange(len(exp_names))
width  = 0.18
colors = ['#1A56DB','#E74C3C','#27AE60','#F39C12']
for i,mname in enumerate(model_names):
    aucs = [exp_results[e][mname]['auc'] for e in exp_names]
    bars = ax.bar(x+i*width, aucs, width,
                  label=mname, color=colors[i], alpha=0.85)
    for bar,v in zip(bars,aucs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f'{v:.3f}', ha='center',
                va='bottom', fontsize=7, rotation=45)
ax.set_xticks(x+width*1.5)
ax.set_xticklabels(exp_names, fontsize=11)
ax.set_ylim(0,1.1)
ax.set_ylabel('AUC-ROC Score', fontsize=12)
ax.set_title('QVulnDetect — AUC by Feature Set and Model',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(y=0.5, color='red', linestyle='--',
           alpha=0.5, label='Random Baseline')
plt.tight_layout()
plt.savefig('results/experiment_auc_comparison.png', dpi=150)
print("\n✅ results/experiment_auc_comparison.png")

# ── Confusion matrix for best ────────────────────────────
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(best_info['r']['cm'],annot=True,fmt='d',
            cmap='Blues',
            xticklabels=['Safe','Vulnerable'],
            yticklabels=['Safe','Vulnerable'],ax=ax)
ax.set_title(f"Best Model: {best_info['model']}\n"
             f"{best_info['exp']}", fontweight='bold')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/best_confusion_matrix.png', dpi=150)
print("✅ results/best_confusion_matrix.png")

print("\n" + "="*60)
print("RESEARCH FINDINGS SUMMARY")
print("="*60)
a_best = max(exp_results['A: Bytecode Only'].values(),
             key=lambda r: r['auc'])['auc']
b_best = max(exp_results['B: Source Only'].values(),
             key=lambda r: r['auc'])['auc']
c_best = max(exp_results['C: Bytecode+Source'].values(),
             key=lambda r: r['auc'])['auc']
print(f"Exp A (Bytecode only)    best AUC: {a_best:.4f}")
print(f"Exp B (Source only)      best AUC: {b_best:.4f}")
print(f"Exp C (Bytecode+Source)  best AUC: {c_best:.4f}")
print(f"\nImprovement A→B : +{(b_best-a_best)*100:.1f}% AUC")
print(f"Improvement A→C : +{(c_best-a_best)*100:.1f}% AUC")
print("\n✅ All done! Check results/ for charts.")
