import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pingouin as pg

df = pd.read_csv("/tmp/gait_final_v8.csv")

plist_names = ["Sebastian","Angela","Cristina","Sofia","Samuel","Valentina","Manya"]
name_to_id = {name: f"P{i+1}" for i, name in enumerate(plist_names)}
df["participant_id"] = df["participant"].map(name_to_id)
plist = [f"P{i+1}" for i in range(7)]

palette = {
    "P1": "#2a78d6", "P2": "#1baf7a", "P3": "#eda100", "P4": "#4a3aa7",
    "P5": "#e34948", "P6": "#eb6834", "P7": "#e87ba4",
}

b  = df["bio_stride_mean"].values; n = df["nor_stride_mean"].values
d  = b - n;  m  = (b + n) / 2
bc = df["bio_stride_cov"].values; nc = df["nor_stride_cov"].values
dc = bc - nc; mc = (bc + nc) / 2

bias_st = np.mean(d); sd_st = np.std(d, ddof=1)
bias_cv = np.mean(dc); sd_cv = np.std(dc, ddof=1)
loa_st_u = bias_st + 1.96*sd_st; loa_st_l = bias_st - 1.96*sd_st
loa_cv_u = bias_cv + 1.96*sd_cv; loa_cv_l = bias_cv - 1.96*sd_cv

t_st, p_st = stats.ttest_rel(b, n)
t_cv, p_cv = stats.ttest_rel(bc, nc)

ba_st = sm.OLS(d,  sm.add_constant(m)).fit()
ba_cv = sm.OLS(dc, sm.add_constant(mc)).fit()

colors = [palette[p] for p in df["participant_id"].values]
participant_legend = [mpatches.Patch(color=palette[p], label=p) for p in plist]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 11,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "xtick.labelsize": 10, "ytick.labelsize": 10,
})

def save(fig, name):
    fig.savefig(f"/mnt/user-data/outputs/{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(f"/mnt/user-data/outputs/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {name}")

# ── Fig 1: BA stride time ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
ax.axhline(bias_st,  color="#2a78d6", linewidth=1.5)
ax.axhline(loa_st_u, color="#888780", linewidth=1.0, linestyle="--")
ax.axhline(loa_st_l, color="#888780", linewidth=1.0, linestyle="--")
ax.axhline(0,        color="#c3c2b7", linewidth=0.8, linestyle=":")
xpad = (m.max()-m.min())*0.04
xline = np.linspace(m.min()-xpad, m.max()+xpad*6, 100)
yline = ba_st.params[0] + ba_st.params[1]*xline
ax.plot(xline, yline, color="#e34948", linewidth=1.3, linestyle="-.")
for mx, dy, c in zip(m, d, colors):
    ax.scatter(mx, dy, color=c, s=60, zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xlim(m.min()-xpad, m.max()+xpad*6)
y_off = 0.003
for val, lbl, col in [(bias_st, f"{bias_st:.4f} s", "#2a78d6"),
                       (loa_st_u, f"{loa_st_u:.4f} s", "#888780"),
                       (loa_st_l, f"{loa_st_l:.4f} s", "#888780")]:
    ax.annotate(lbl, xy=(m.max()+xpad*0.6, val+y_off), fontsize=9, color=col, va="bottom")
ax.set_xlabel("Mean stride time — (BIOPAC + Noraxon) / 2 (s)", fontsize=10, color="#52514e")
ax.set_ylabel("Difference — BIOPAC − Noraxon (s)", fontsize=10, color="#52514e")
ax.set_title("Bland-Altman: stride time", fontsize=12, fontweight="bold", pad=10, color="#0b0b0b")
ax.yaxis.grid(True); ax.set_axisbelow(True)
reg_line = mlines.Line2D([], [], color="#e34948", linewidth=1.3, linestyle="-.",
                          label=f"Regression (slope={ba_st.params[1]:.3f}, p={ba_st.pvalues[1]:.3f})")
ax.legend(handles=participant_legend+[reg_line], fontsize=8.5, loc="upper left",
          framealpha=0.9, edgecolor="#c3c2b7", title="Participant", title_fontsize=9)
plt.tight_layout()
save(fig, "fig1_ba_stride_time")

# ── Fig 2: BA stride CoV ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
ax.axhline(bias_cv,  color="#2a78d6", linewidth=1.5)
ax.axhline(loa_cv_u, color="#888780", linewidth=1.0, linestyle="--")
ax.axhline(loa_cv_l, color="#888780", linewidth=1.0, linestyle="--")
ax.axhline(0,        color="#c3c2b7", linewidth=0.8, linestyle=":")
xpad_c = (mc.max()-mc.min())*0.04
xline_c = np.linspace(mc.min()-xpad_c, mc.max()+xpad_c*7, 100)
yline_c = ba_cv.params[0] + ba_cv.params[1]*xline_c
ax.plot(xline_c, yline_c, color="#e34948", linewidth=1.3, linestyle="-.")
for mx, dy, c in zip(mc, dc, colors):
    ax.scatter(mx, dy, color=c, s=60, zorder=3, edgecolors="white", linewidths=0.5)
ax.set_xlim(mc.min()-xpad_c, mc.max()+xpad_c*7)
y_off_c = 0.3
for val, lbl, col in [(bias_cv, f"{bias_cv:.2f}%", "#2a78d6"),
                       (loa_cv_u, f"{loa_cv_u:.2f}%", "#888780"),
                       (loa_cv_l, f"{loa_cv_l:.2f}%", "#888780")]:
    ax.annotate(lbl, xy=(mc.max()+xpad_c*0.6, val+y_off_c), fontsize=9, color=col, va="bottom")
ax.set_xlabel("Mean stride time CoV — (BIOPAC + Noraxon) / 2 (%)", fontsize=10, color="#52514e")
ax.set_ylabel("Difference — BIOPAC − Noraxon (%)", fontsize=10, color="#52514e")
ax.set_title("Bland-Altman: stride time CoV", fontsize=12, fontweight="bold", pad=10, color="#0b0b0b")
ax.yaxis.grid(True); ax.set_axisbelow(True)
reg_line_c = mlines.Line2D([], [], color="#e34948", linewidth=1.3, linestyle="-.",
                            label=f"Regression (slope={ba_cv.params[1]:.3f}, p={ba_cv.pvalues[1]:.3f})")
ax.legend(handles=participant_legend+[reg_line_c], fontsize=8.5, loc="upper left",
          framealpha=0.9, edgecolor="#c3c2b7", title="Participant", title_fontsize=9)
plt.tight_layout()
save(fig, "fig2_ba_stride_cov")

# ── Boxplots ──────────────────────────────────────────────────────────────────
bio_patch = mpatches.Patch(facecolor="#888780", label="BIOPAC")
nor_patch  = mpatches.Patch(facecolor="white", edgecolor="#888780", hatch="///", label="Noraxon")

def make_boxplot(fname, value_key, ylabel, title):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    box_data = []; box_colors_list = []
    for p_id, p_name in zip(plist, plist_names):
        sub = df[df.participant == p_name]
        box_data.append(sub[f"bio_{value_key}"].values)
        box_data.append(sub[f"nor_{value_key}"].values)
        box_colors_list += [palette[p_id], "white"]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#0b0b0b", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8, color="#52514e"),
                    capprops=dict(linewidth=0.8, color="#52514e"),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="#888780", markeredgecolor="#888780"))
    for patch, fc, p_id in zip(bp["boxes"], box_colors_list,
                                 [p for p in plist for _ in range(2)]):
        patch.set_facecolor(fc); patch.set_edgecolor(palette[p_id]); patch.set_linewidth(1.2)
        if fc == "white": patch.set_hatch("///"); patch.set_alpha(0.8)
    tick_positions = [i*2 + 1.5 for i in range(len(plist))]
    ax.set_xticks(tick_positions); ax.set_xticklabels(plist, fontsize=10)
    ax.set_xlabel("Participant", fontsize=10, color="#52514e")
    ax.set_ylabel(ylabel, fontsize=10, color="#52514e")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#0b0b0b")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for i in range(1, len(plist)):
        ax.axvline(i*2 + 0.5, color="#e1e0d9", linewidth=0.8)
    ax.legend(handles=[bio_patch, nor_patch], fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="#c3c2b7")
    plt.tight_layout()
    save(fig, fname)

make_boxplot("fig3_boxplot_stride_time", "stride_mean", "Stride time (s)", "Stride time by participant")
make_boxplot("fig4_boxplot_stride_cov",  "stride_cov",  "Stride time CoV (%)", "Stride time CoV by participant")

# ── Per-trial CSV ─────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "Participant": df["participant_id"], "Trial": df["trial"],
    "BIOPAC strides (n)": df["n_strides_bio"], "Noraxon strides (n)": df["n_strides_nor"],
    "BIOPAC stride mean (s)": df["bio_stride_mean"].round(4),
    "BIOPAC stride SD (s)": df["bio_stride_sd"].round(4),
    "BIOPAC stride CoV (%)": df["bio_stride_cov"].round(2),
    "Noraxon stride mean (s)": df["nor_stride_mean"].round(4),
    "Noraxon stride SD (s)": df["nor_stride_sd"].round(4),
    "Noraxon stride CoV (%)": df["nor_stride_cov"].round(2),
    "Difference — stride mean (s)": df["diff_stride"].round(4),
    "Difference — stride CoV (%)": df["diff_cov"].round(2),
}).sort_values(["Participant","Trial"]).reset_index(drop=True)
out.to_csv("/mnt/user-data/outputs/per_trial_results.csv", index=False)

# ── Mixed-effects models ──────────────────────────────────────────────────────
long_rows = []
for _, r in df.iterrows():
    long_rows.append({"participant": r.participant_id, "system": "BIOPAC",  "stride_time": r.bio_stride_mean, "cov": r.bio_stride_cov})
    long_rows.append({"participant": r.participant_id, "system": "Noraxon", "stride_time": r.nor_stride_mean, "cov": r.nor_stride_cov})
long_df = pd.DataFrame(long_rows)
long_df["system"] = pd.Categorical(long_df["system"], categories=["Noraxon","BIOPAC"])

model_st = smf.mixedlm("stride_time ~ system", long_df, groups=long_df["participant"]).fit(reml=True)
model_cv = smf.mixedlm("cov ~ system", long_df, groups=long_df["participant"]).fit(reml=True)

# ── ICC ──────────────────────────────────────────────────────────────────────
def run_icc_a1(bv, nv):
    nn=len(bv)
    long=pd.DataFrame({"target":list(range(nn))*2,"rater":["BIOPAC"]*nn+["Noraxon"]*nn,"value":list(bv)+list(nv)})
    icc=pg.intraclass_corr(data=long,targets="target",raters="rater",ratings="value")
    row=icc[icc.Type=="ICC(A,1)"].iloc[0]
    return row["ICC"], row["CI95"][0], row["CI95"][1], row["pval"]
icc_st = run_icc_a1(b,n)
icc_cv = run_icc_a1(bc,nc)

# ── Participant-level paired t-test ───────────────────────────────────────────
per_p = df.groupby("participant").agg(
    bio_stride=("bio_stride_mean","mean"), nor_stride=("nor_stride_mean","mean"),
    bio_cov=("bio_stride_cov","mean"),     nor_cov=("nor_stride_cov","mean"),
).reset_index()
t_st_p, p_st_p = stats.ttest_rel(per_p.bio_stride, per_p.nor_stride)
t_cv_p, p_cv_p = stats.ttest_rel(per_p.bio_cov,    per_p.nor_cov)

summary = pd.DataFrame({
    "Parameter":                  ["Stride time (s)", "Stride CoV (%)"],
    "BIOPAC mean":                [f"{np.mean(b):.4f}",  f"{np.mean(bc):.2f}"],
    "BIOPAC SD":                  [f"{np.std(b,ddof=1):.4f}", f"{np.std(bc,ddof=1):.2f}"],
    "Noraxon mean":               [f"{np.mean(n):.4f}", f"{np.mean(nc):.2f}"],
    "Noraxon SD":                 [f"{np.std(n,ddof=1):.4f}", f"{np.std(nc,ddof=1):.2f}"],
    "Bias (B-N)":                 [f"{bias_st:.4f}", f"{bias_cv:.2f}"],
    "SD of diff":                 [f"{sd_st:.4f}", f"{sd_cv:.2f}"],
    "LoA lower":                  [f"{loa_st_l:.4f}", f"{loa_cv_l:.2f}"],
    "LoA upper":                  [f"{loa_st_u:.4f}", f"{loa_cv_u:.2f}"],
    "Paired t (trial-level)":     [f"{t_st:.4f}", f"{t_cv:.4f}"],
    "Paired p (trial-level)":     [f"{p_st:.4f}", f"{p_cv:.4f}"],
    "Paired t (participant-level)": [f"{t_st_p:.4f}", f"{t_cv_p:.4f}"],
    "Paired p (participant-level)": [f"{p_st_p:.4f}", f"{p_cv_p:.4f}"],
    "ICC(A,1)":                   [f"{icc_st[0]:.4f}", f"{icc_cv[0]:.4f}"],
    "ICC 95% CI lower":           [f"{icc_st[1]:.4f}", f"{icc_cv[1]:.4f}"],
    "ICC 95% CI upper":           [f"{icc_st[2]:.4f}", f"{icc_cv[2]:.4f}"],
    "ICC p-value":                [f"{icc_st[3]:.6f}", f"{icc_cv[3]:.6f}"],
    "Mixed-model system effect (β)": [f"{model_st.params['system[T.BIOPAC]']:.4f}", f"{model_cv.params['system[T.BIOPAC]']:.4f}"],
    "Mixed-model SE":             [f"{model_st.bse['system[T.BIOPAC]']:.4f}", f"{model_cv.bse['system[T.BIOPAC]']:.4f}"],
    "Mixed-model p-value":        [f"{model_st.pvalues['system[T.BIOPAC]']:.4f}", f"{model_cv.pvalues['system[T.BIOPAC]']:.4f}"],
    "Mixed-model participant variance": [f"{model_st.cov_re.iloc[0,0]:.6f}", f"{model_cv.cov_re.iloc[0,0]:.6f}"],
    "BA regression slope": [f"{ba_st.params[1]:.4f}", f"{ba_cv.params[1]:.4f}"],
    "BA regression slope p-value": [f"{ba_st.pvalues[1]:.4f}", f"{ba_cv.pvalues[1]:.4f}"],
})
summary.to_csv("/mnt/user-data/outputs/stats_summary.csv", index=False)

print("All outputs regenerated (n=19 trials, Cristina t3 excluded).")
print(summary.T.to_string())
