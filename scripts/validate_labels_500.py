import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/processed/quantum_labels.csv')
SOURCE_DIR = 'data/raw/Source Codes'

df['quantum_broken'] = (
    (df['Q1_ecrecover_detected']==1) |
    (df['Q2_oz_ecdsa_detected']==1)  |
    (df['Q3_sig_checker_detected']==1)
).astype(int)

# 250 positives + 250 negatives = 500 total
pos = df[df['quantum_broken']==1].sample(
    min(250, df['quantum_broken'].sum()), random_state=99)
neg = df[df['quantum_broken']==0].sample(250, random_state=99)
sample = pd.concat([pos,neg]).sample(
    frac=1, random_state=99).reset_index(drop=True)

PATTERNS = ['ecrecover(','ECDSA.recover(','SignatureChecker.']
correct = 0
errors  = []

for _, row in sample.iterrows():
    fpath = os.path.join(SOURCE_DIR, row['category'],
                         row['category'], row['filename'])
    if not os.path.exists(fpath):
        fpath = os.path.join(SOURCE_DIR,
                 row['category'], row['filename'])
    try:
        src = open(fpath,'r',errors='ignore').read()
    except:
        src = ""

    ground_truth = int(any(p in src for p in PATTERNS))
    label        = row['quantum_broken']

    if ground_truth == label:
        correct += 1
    else:
        errors.append({
            'file':row['filename'][:30],
            'label':label,
            'actual':ground_truth,
            'category':row['category']
        })

accuracy = correct/len(sample)*100
print(f"Sample size    : {len(sample)}")
print(f"Correct labels : {correct}/{len(sample)} ({accuracy:.1f}%)")
print(f"Label errors   : {len(errors)}")

import scipy.stats as stats
ci_low  = stats.binom.ppf(0.025, len(sample), correct/len(sample))/len(sample)*100
ci_high = stats.binom.ppf(0.975, len(sample), correct/len(sample))/len(sample)*100
print(f"95% CI         : [{ci_low:.1f}%, {ci_high:.1f}%]")

if errors:
    print("\nErrors found:")
    for e in errors[:10]:
        print(f"  {e['file']}  label={e['label']} actual={e['actual']}")

pd.DataFrame(errors if errors else [{'result':'all correct'}]
).to_csv('results/label_validation_500.csv', index=False)
print("\nSaved: results/label_validation_500.csv")
