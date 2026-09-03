"""
Compute trial-level stride time statistics comparing BIOPAC BN-STRIKE vs
Noraxon MyoMotion, for the dataset with Cristina's Trial 3 excluded
(n = 19 trials, 7 participants).
 
Data pipeline:
  1. BIOPAC: heel strikes detected from CH13 (heel sensor) using a
     60%-of-peak amplitude threshold on the raw voltage signal, with a
     minimum-duration filter to remove short spurious on/off flickers,
     then trimmed to drop non-physiological leading/trailing intervals.
  2. Noraxon: heel strikes detected as rising edges (0 -> 1000) in the
     Foot_RT-Contact.csv signal exported from the MyoMotion software.
  3. Both heel-strike series are time-aligned to each system's own first
     detected heel strike (t = 0), and stride time is the interval
     between consecutive heel strikes.
  4. A paired t-test compares the 19 per-trial mean stride times
     (one BIOPAC value and one Noraxon value per trial).
"""
 
import numpy as np
import pandas as pd
from scipy import stats
 
# ── BIOPAC loading & heel-strike detection ────────────────────────────────
def load_biopac_smart(path):
    """Load a BIOPAC .txt export, auto-detecting whether it has a time
    column, and return (time, ch1, ch13)."""
    with open(path, 'r') as f:
        lines = f.readlines()
    data_start = None
    has_time_col = False
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        try:
            float(parts[0])
            data_start = i
            has_time_col = (float(parts[0]) < 1.0)
            break
        except (ValueError, IndexError):
            continue
 
    df = pd.read_csv(path, sep='\t', skiprows=data_start, header=None).dropna(how='all')
    if has_time_col and df.shape[1] >= 3:
        df.columns = ['time', 'ch1', 'ch13'] + [f'x{i}' for i in range(df.shape[1] - 3)]
        for c in ['time', 'ch1', 'ch13']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['time', 'ch1', 'ch13'])
        return df['time'].values, df['ch1'].values, df['ch13'].values
    else:
        df.columns = ['ch1', 'ch13'] + [f'x{i}' for i in range(df.shape[1] - 2)]
        for c in ['ch1', 'ch13']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['ch1', 'ch13'])
        time = np.arange(len(df)) * 0.0005  # 2000 Hz sampling assumed
        return time, df['ch1'].values, df['ch13'].values
 
 
def get_heel_strikes(time, ch13, cutoff=None):
    """Detect heel strikes as rising edges of CH13 crossing 60% of its
    peak amplitude, with a minimum-duration filter to suppress noise."""
    if cutoff is not None:
        mask = time <= cutoff
        time, ch13 = time[mask], ch13[mask]
    if len(ch13) == 0 or np.max(ch13) < 0.5:
        return np.array([])
    if np.min(ch13) < -1.0:
        ch13 = np.abs(ch13)
 
    threshold = 0.6 * np.max(ch13)
    contact = (ch13 > threshold).astype(int)
    dt = np.median(np.diff(time))
    dt = dt if dt > 0 else 0.0005
    fs = 1.0 / dt
 
    clean = contact.copy()
    d = np.diff(clean)
    rise_idx = np.where(d == 1)[0] + 1
    fall_idx = np.where(d == -1)[0] + 1
    for r in rise_idx:
        f = fall_idx[fall_idx > r]
        if len(f) == 0:
            break
        if (f[0] - r) < int(0.1 * fs):   # drop stance shorter than 0.1 s
            clean[r:f[0]] = 0
 
    d2 = np.diff(clean)
    rise2 = np.where(d2 == 1)[0] + 1
    fall2 = np.where(d2 == -1)[0] + 1
    for f in fall2:
        r = rise2[rise2 > f]
        if len(r) == 0:
            break
        if (r[0] - f) < int(0.2 * fs):   # drop swing shorter than 0.2 s
            clean[f:r[0]] = 1
 
    d3 = np.diff(clean)
    return time[1:][d3 == 1]
 
 
def trim_hs(hs, min_int=0.7, max_int=2.2):
    """Drop leading/trailing heel strikes whose adjacent interval falls
    outside a physiologically plausible walking-stride range."""
    if len(hs) < 2:
        return hs
    changed = True
    while changed and len(hs) >= 2:
        changed = False
        intervals = np.diff(hs)
        if intervals[0] < min_int or intervals[0] > max_int:
            hs = hs[1:]; changed = True; continue
        if intervals[-1] < min_int or intervals[-1] > max_int:
            hs = hs[:-1]; changed = True; continue
    return hs
 
 
# ── Noraxon loading & heel-strike detection ───────────────────────────────
def load_noraxon(path):
    """Load a Foot_RT-Contact.csv export and return (heel_strikes, toe_offs)
    as rising/falling edges of the 0/1000 contact signal."""
    df = pd.read_csv(path, skiprows=4)
    df.columns = ['time', 'value']
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna()
    contact_bin = (df['value'].values >= 500).astype(int)
    d = np.diff(contact_bin)
    t = df['time'].values
    return t[1:][d == 1], t[1:][d == -1]
 
 
def compute_gait(hs, to):
    """Pair each heel strike with the next toe-off and the following heel
    strike to compute stance, swing, and stride durations."""
    stances, swings, strides = [], [], []
    for i, h in enumerate(hs[:-1]):
        ta = to[to > h]
        if len(ta) == 0:
            continue
        t = ta[0]
        nh = hs[i + 1]
        if t >= nh:
            continue
        stances.append(t - h)
        swings.append(nh - t)
        strides.append(nh - h)
    return np.array(stances), np.array(swings), np.array(strides)
 
 
def cov(a):
    return (np.std(a, ddof=1) / np.mean(a)) * 100 if len(a) > 1 else np.nan
 
 
# ── Participant / trial file map — Cristina Trial 3 EXCLUDED ─────────────
participants_cfg = {
    "Sebastian": {1: "/mnt/user-data/uploads/Sebastian_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Sebastian_20m_test2real.txt",
                  3: "/mnt/user-data/uploads/Sebastian_20m_test3real.txt"},
    "Angela":    {1: "/mnt/user-data/uploads/Angela_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Angela_20m_test2real.txt",
                  3: "/mnt/user-data/uploads/Angela_20m_test3real.txt"},
    "Cristina":  {1: "/mnt/user-data/uploads/Cristina_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Cristina_20m_test2real_trimmed.txt"},
                  # Trial 3 intentionally excluded
    "Sofia":     {1: "/mnt/user-data/uploads/Sofia_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Sofia_20m_test2real.txt",
                  3: "/mnt/user-data/uploads/Sofia_20m_test3real.txt"},
    "Samuel":    {1: "/mnt/user-data/uploads/Samuel_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Samuel_20m_test2real.txt",
                  3: "/mnt/user-data/uploads/Samuel_20m_test3real.txt"},
    "Valentina": {1: "/mnt/user-data/uploads/Valentina_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Valentina_20m_test2real.txt",
                  3: "/mnt/user-data/uploads/Valentina_20m_test3real.txt"},
    "Manya":     {1: "/mnt/user-data/uploads/Manya_20m_test1real.txt",
                  2: "/mnt/user-data/uploads/Manya_20m_test2real.txt"},
}
 
all_rows = []
for name, trials in participants_cfg.items():
    for t, bpath in trials.items():
        npath = f"/tmp/noraxon_{name}_t{t}.csv"
 
        time, ch1, ch13 = load_biopac_smart(bpath)
        b_hs_raw = get_heel_strikes(time, ch13)
        if len(b_hs_raw) < 2:
            continue
        b_hs_raw = trim_hs(b_hs_raw)
        if len(b_hs_raw) < 2:
            continue
        b_hs = b_hs_raw - b_hs_raw[0]
        b_strides = np.diff(b_hs)
 
        n_hs_raw, n_to_raw = load_noraxon(npath)
        n_hs = n_hs_raw - n_hs_raw[0]
        n_to = n_to_raw - n_hs_raw[0]
        n_st, n_sw, n_str = compute_gait(n_hs, n_to)
 
        all_rows.append({
            "participant": name, "trial": t,
            "bio_stride_mean": np.mean(b_strides),
            "bio_stride_sd":   np.std(b_strides, ddof=1),
            "bio_stride_cov":  cov(b_strides),
            "nor_stride_mean": np.mean(n_str),
            "nor_stride_sd":   np.std(n_str, ddof=1),
            "nor_stride_cov":  cov(n_str),
        })
 
df = pd.DataFrame(all_rows)
print(f"Total trials: {len(df)}")  # should be 19
 
# ── Trial-level paired t-test ──────────────────────────────────────────────
b = df["bio_stride_mean"].values
n = df["nor_stride_mean"].values
 
t_stat, p_val = stats.ttest_rel(b, n)
 
print("\n=== TRIAL-LEVEL STRIDE TIME (n=19 trials) ===")
print(f"BIOPAC  — mean: {b.mean():.4f} s, SD: {b.std(ddof=1):.4f} s")
print(f"Noraxon — mean: {n.mean():.4f} s, SD: {n.std(ddof=1):.4f} s")
print(f"Paired t-test: t({len(b)-1}) = {t_stat:.3f}, p = {p_val:.3f}")
 
