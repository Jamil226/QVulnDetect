import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

print("="*60)
print("QVulnDetect — TF-IDF Final (15000 char limit)")
print("="*60)

# ── Load labels ───────────────────────────────────────────
print("\n[1/6] Loading labels...")
df = pd.read_csv('data/processed/quantum_labels.csv')
total = len(df)

df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1) |
    (df['Q2_oz_ecdsa_detected']==1)  |
    (df['Q3_sig_checker_detected']==1)
).astype(int)

qb = df['quantum_broken'].sum()
print(f"Quantum-broken : {qb:,} ({qb/total*100:.1f}%)")
print(f"Quantum-safe   : {total-qb:,} ({(total-qb)/total*100:.1f}%)")

# ── Load source code ──────────────────────────────────────
print("\n[2/6] Loading source code (15000 chars)...")
SOURCE_DIR = 'data/raw/Source Codes'

texts  = []
labels = []
for _, row in tqdm(df.iterrows(), total=total, desc="Reading"):
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR, row['category'],
                             row['filename'])
    try:
        text = open(fpath,'r',errors='ignore').read(15000)
    except:
        text = ""
    texts.append(text)
    labels.append(row['quantum_broken'])

y = np.array(labels)
print(f"Loaded {len(texts):,} files")

# Verify ecrecover is now captured
sample_vuln = df[df['quantum_broken']==1].index[:100]
found_in_text = sum(1 for i in sample_vuln if 'ecrecover' in texts[i])
print(f"ecrecover found in first 100 vulnerable contracts: {found_in_text}/100")

# ── TF-IDF ────────────────────────────────────────────────
print("\n[3/6] TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'[a-zA-Z_][a-zA-Z0-9_]*',
    max_features=2000,
    min_df=5,
    max_df=0.95,
    ngram_range=(1,2),
    sublinear_tf=True,
)
X = vectorizer.fit_transform(texts)
print(f"Matrix shape : {X.shape}")
print(f"Memory       : {X.data.nbytes/1024/1024:.1f} MB")

vocab = vectorizer.get_feature_names_out()
for t in ['ecrecover','ECDSA','recover','secp256k1',
          'blockhash','timestamp']:
    status = 'FOUND ✅' if t in vocab else 'not found ❌'
    print(f"  {t:30}: {status}")

# ── Split ─────────────────────────────────────────────────
print("\n[4/6] Splitting (80/20)...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y)
print(f"Train:{X_tr.shape[0]:,}  Test:{X_te.shape[0]:,}")
print(f"Test positive: {y_te.sum():,} ({y_te.mean()*100:.1f}%)")

# ── Train ─────────────────────────────────────────────────
print("\n[5/6] Training models...\n")
os.makedirs('results', exist_ok=True)

models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, class_weight='balanced',
        max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=20,
        class_weight='balanced',
        random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"  ▶ {name}...")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]
    elapsed = time.time()-t0

    results[name] = {
        'acc':  accuracy_score(y_te, y_pred),
        'f1':   f1_score(y_te, y_pred, zero_division=0),
        'prec': precision_score(y_te, y_pred, zero_division=0),
        'rec':  recall_score(y_te, y_pred, zero_division=0),
        'auc':  roc_auc_score(y_te, y_prob),
        'cm':   confusion_matrix(y_te, y_pred),
        'time': elapsed,
    }
    r = results[name]
    print(f"    Acc:{r['acc']:.4f}  F1:{r['f1']:.4f}  "
          f"Prec:{r['prec']:.4f}  Rec:{r['rec']:.4f}  "
          f"AUC:{r['auc']:.4f}  ({elapsed:.1f}s)\n")

# ── Summary ───────────────────────────────────────────────
print("="*60)
print("FINAL RESULTS")
print("="*60)
print(f"{'Model':<25}{'Acc':>8}{'F1':>8}"
      f"{'Prec':>8}{'Rec':>8}{'AUC':>8}")
print("-"*60)
print(f"{'Bytecode ML (Exp A)':<25}{'—':>8}{'—':>8}"
      f"{'—':>8}{'—':>8}{'0.5197':>8}")
for n,r in results.items():
    print(f"{n:<25}{r['acc']:>8.4f}{r['f1']:>8.4f}"
          f"{r['prec']:>8.4f}{r['rec']:>8.4f}"
          f"{r['auc']:>8.4f}")

best_name = max(results, key=lambda n: results[n]['auc'])
best = results[best_name]
print(f"\n🏆 Best : {best_name}")
print(f"   AUC  : {best['auc']:.4f}")
print(f"   Improvement over bytecode: "
      f"+{(best['auc']-0.5197)*100:.1f}%")

# ── Top tokens ────────────────────────────────────────────
print("\n[6/6] Top predictive tokens (Logistic Regression)...")
lr    = models['Logistic Regression']
coefs = lr.coef_[0]
top20 = np.argsort(coefs)[-20:][::-1]
print("\nTop 20 tokens → QUANTUM VULNERABLE:")
for i in top20:
    print(f"  {vocab[i]:<35} {coefs[i]:.4f}")

# ── Plot 1: AUC comparison ────────────────────────────────
fig, ax = plt.subplots(figsize=(9,5))
names  = ['Bytecode ML\n(AUC≈0.51\nrandom)'] + list(results.keys())
aucs   = [0.5197] + [results[n]['auc'] for n in results]
colors = ['#95A5A6','#1A56DB','#E74C3C']
bars   = ax.bar(range(len(names)), aucs,
                color=colors, alpha=0.85, width=0.4)
for bar,v in zip(bars,aucs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.008,
            f'{v:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='red', linestyle='--',
           linewidth=2, alpha=0.7, label='Random baseline (0.50)')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=11)
ax.set_ylim(0.3, 1.1)
ax.set_ylabel('AUC-ROC', fontsize=13)
ax.set_title('Bytecode Features vs TF-IDF Source Code\n'
             'Quantum-Broken Detection (ecrecover/ECDSA)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('results/tfidf_auc_final.png', dpi=150)
print("\n✅ results/tfidf_auc_final.png")

# ── Plot 2: Confusion matrix ──────────────────────────────
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(best['cm'], annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe','Q-Broken'],
            yticklabels=['Safe','Q-Broken'], ax=ax)
ax.set_title(f'Confusion Matrix\n{best_name}',
             fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/tfidf_confusion_final.png', dpi=150)
print("✅ results/tfidf_confusion_final.png")

# ── Plot 3: Fixed taxonomy ────────────────────────────────
qa = int(((df['Q5_timestamp_prng_detected']==1) |
          (df['Q6_blockhash_prng_detected']==1)).sum())
fig, ax = plt.subplots(figsize=(10,4))
cats   = ['Quantum-Broken\n(ECDSA/ecrecover)\nShor directly applies',
          'Quantum-Amplified\n(PRNG patterns)\nClassical+quantum risk',
          'Quantum-Safe\nNo pattern detected']
counts = [qb, qa, total-qb-qa]
colors_b = ['#E74C3C','#F39C12','#27AE60']
bars   = ax.bar(cats, counts, color=colors_b, alpha=0.85)
for bar,v in zip(bars,counts):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+200,
            f'{v:,}\n({v/total*100:.1f}%)',
            ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Contracts', fontsize=12)
ax.set_title('Fixed QVPT — Honest Taxonomy\n111,897 Smart Contracts',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(counts)*1.2)
plt.tight_layout()
plt.savefig('results/fixed_taxonomy_final.png', dpi=150)
print("✅ results/fixed_taxonomy_final.png")

pd.DataFrame([
    {'Model':n,'Accuracy':r['acc'],'F1':r['f1'],
     'Precision':r['prec'],'Recall':r['rec'],'AUC':r['auc']}
    for n,r in results.items()
]).to_csv('results/tfidf_final_results.csv', index=False)
print("✅ results/tfidf_final_results.csv")
print("\n✅ All done!")
