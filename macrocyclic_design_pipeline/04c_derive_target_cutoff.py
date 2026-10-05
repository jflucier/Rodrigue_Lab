#!/usr/bin/env python3
import pandas as pd
import numpy as np

# 1. Load your generated data pool
csv_file = "afcyc_filter_summary.csv"
try:
    df = pd.read_csv(csv_file)
except Exception:
    print(f"Could not read {csv_file}. Ensure you have generated scoring records.")
    exit(1)

print(f"Loaded {len(df)} design predictions for your new target.")

# 2. Filter out catastrophically failed shapes first (RMSD > 2.0 or extreme iPAE)
valid_pool = df[df["rmsd"] < 2.0]

if valid_pool.empty:
    print("[WARN] No designs passed a basic structural RMSD < 2.0 A filter.")
    exit(1)

# 3. Calculate target-specific distribution percentiles
# The authors picked cutoffs that isolate the top high-confidence binders
ipae_array = valid_pool["normalized_ipae"].to_numpy()

median_ipae = np.median(ipae_array)
top_25_percentile = np.percentile(ipae_array, 25)
top_10_percentile = np.percentile(ipae_array, 10)

print("\n" + "="*50)
print("📊 MATHEMATICAL iPAE DISTRIBUTION FOR YOUR TARGET")
print("="*50)
print(f"Median Normalized iPAE       : {median_ipae:.3f}  (Raw: {median_ipae*32:.1f})")
print(f"Top 25% High-Affinity Binders : {top_25_percentile:.3f}  (Raw: {top_25_percentile*32:.1f})")
print(f"Top 10% Elite Binder Designs : {top_10_percentile:.3f}  (Raw: {top_10_percentile*32:.1f})")
print("="*50)

# 4. Provide the definitive recommendations based on the paper's distribution logic
print("\n💡 HOW TO PICK YOUR PRODUCTION CUTOFF:")
if top_25_percentile <= 0.20:
    print(f"-> Your target binds easily! Use a strict cutoff of: 0.20 (Similar to MCL1)")
elif top_25_percentile <= 0.30:
    print(f"-> Standard target profile. Use a moderate cutoff of: {top_25_percentile:.2f} or 0.30 (Similar to MDM2)")
else:
    print(f"-> Tough/Difficult pocket target! Use a relaxed cutoff of: {min(0.40, median_ipae):.2f} (Similar to RbtA)")
