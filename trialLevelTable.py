###----We did not use per-participant trials for our table in the paper----

import numpy as np, pandas as pd
from scipy import stats

df = pd.read_csv("/tmp/gait_final_v4.csv") #<--- whatever data we gave it, one file (csv) for noraxon and another for biopac

# Per-participant: average across trials first, then compute mean/SD across participants
print("=== PER-PARTICIPANT MEANS (averaged across trials) ===")
per_p = df.groupby("participant").agg(
    bio_stride=("bio_stride_mean","mean"),
    nor_stride=("nor_stride_mean","mean"),
).reset_index()
print(per_p.to_string(index=False))

# Mean and SD across participants
print(f"\nBIOPAC  — mean across participants: {per_p.bio_stride.mean():.4f} s, SD: {per_p.bio_stride.std(ddof=1):.4f} s")
print(f"Noraxon — mean across participants: {per_p.nor_stride.mean():.4f} s, SD: {per_p.nor_stride.std(ddof=1):.4f} s")

# Paired t-test at participant level
t_p, p_p = stats.ttest_rel(per_p.bio_stride, per_p.nor_stride)
print(f"\nPaired t-test (participant level, n=6): t={t_p:.4f}, p={p_p:.4f}")

# Also at trial level (n=17)
print("\n=== TRIAL-LEVEL (n=17 trials) ===")
b = df.bio_stride_mean.values; n = df.nor_stride_mean.values
t_t, p_t = stats.ttest_rel(b, n)
print(f"BIOPAC  — mean: {b.mean():.4f} s, SD: {b.std(ddof=1):.4f} s")
print(f"Noraxon — mean: {n.mean():.4f} s, SD: {n.std(ddof=1):.4f} s")
print(f"Paired t-test (trial level, n=17): t={t_t:.4f}, p={p_t:.4f}")

