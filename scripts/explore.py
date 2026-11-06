import pandas as pd

csv_path = 'data/raw/BCCC-SCsVul-2024.csv'
df = pd.read_csv(csv_path)

print("="*50)
print("DATASET OVERVIEW")
print("="*50)
print(f"Rows (contracts)    : {df.shape[0]:,}")
print(f"Columns (features)  : {df.shape[1]}")

print("\n" + "="*50)
print("ALL COLUMN NAMES")
print("="*50)
for i, col in enumerate(df.columns):
    print(f"  {i+1:3}. {col}")

print("\n" + "="*50)
print("LAST 5 COLUMNS - LABEL DISTRIBUTION")
print("="*50)
last5 = df.columns[-5:]
for col in last5:
    print(f"\n  {col}:")
    print(f"    {df[col].value_counts().to_dict()}")
