"""
Generates 4 high-resolution matplotlib figures for the report and presentation slides:
- fig1_macro_regimes.png: Subscription Rate Across Euribor 3M & Employment Variation Regimes
- fig2_poutcome_contact.png: Prior Campaign Outcome & Contact Channel Impact on Conversion
- fig3_campaign_fatigue.png: Campaign Contact Fatigue & Call Duration Leakage Comparison
- fig4_model_comparison.png: Test-Set Macro F1 Comparison with 95% Bootstrap Confidence Intervals
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "bank_marketing.csv")
METRICS = os.path.join(HERE, "results", "metrics.csv")
OUT_DIRS = [
    os.path.join(HERE, "results")
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.8,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

NAVY = "#0F172A"
BLUE = "#1D4ED8"
TEAL = "#0D9488"
AMBER = "#D97706"
CORAL = "#E11D48"
SLATE = "#64748B"
LIGHT_BG = "#F8FAFC"

df = pd.read_csv(DATA)
met = pd.read_csv(METRICS)

def save_both(fig, filename):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, filename), dpi=220, bbox_inches="tight")
    plt.close(fig)

# Figure 1: Macroeconomic Regimes (Euribor 3M buckets & Employment Variation Rate)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
df["euribor_bin"] = pd.cut(
    df["euribor3m"],
    bins=[0, 1.2, 2.0, 4.5, 6.0],
    labels=["< 1.2%\n(Low Rate)", "1.2% - 2.0%\n(Moderate)", "2.0% - 4.5%\n(Transitional)", "> 4.5%\n(Tight Monetary)"]
)
grp1 = df.groupby("euribor_bin", observed=False)["y"].agg(["mean", "count"]).reset_index()
bars1 = ax1.bar(grp1["euribor_bin"].astype(str), grp1["mean"] * 100, color=[TEAL, BLUE, SLATE, NAVY], width=0.58)
ax1.set_title("A. Subscription Rate by Euribor 3-Month Regime")
ax1.set_ylabel("Term Deposit Subscription Rate (%)")
ax1.set_ylim(0, max(grp1["mean"] * 100) * 1.25)
ax1.grid(axis="y")
for b, cnt in zip(bars1, grp1["count"]):
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY)

# Panel B: Job category subscription rates (top 8 categories by volume)
top_jobs = df["job"].value_counts().head(8).index
grp2 = df[df["job"].isin(top_jobs)].groupby("job")["y"].mean().sort_values(ascending=True) * 100
colors2 = [TEAL if v > 18 else (BLUE if v > 11 else SLATE) for v in grp2.values]
bars2 = ax2.barh(grp2.index, grp2.values, color=colors2, height=0.62)
ax2.set_title("B. Subscription Rate Across Client Occupations")
ax2.set_xlabel("Term Deposit Subscription Rate (%)")
ax2.set_xlim(0, max(grp2.values) * 1.22)
ax2.grid(axis="x")
for b in bars2:
    w = b.get_width()
    ax2.text(w + 0.5, b.get_y() + b.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color=NAVY)

fig.tight_layout(pad=2.0)
save_both(fig, "fig1_macro_regimes.png")

# Figure 2: Prior Campaign Outcome (poutcome) & Contact Channel Interaction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
pout_order = ["nonexistent", "failure", "success"]
piv = df.pivot_table(index="poutcome", columns="contact", values="y", aggfunc="mean").loc[pout_order] * 100
x = np.arange(len(pout_order))
w = 0.35
b_cell = ax1.bar(x - w/2, piv["cellular"], width=w, label="Cellular", color=BLUE)
b_tel = ax1.bar(x + w/2, piv["telephone"], width=w, label="Telephone (Landline)", color=SLATE)
ax1.set_xticks(x)
ax1.set_xticklabels(["Nonexistent\n(pdays = 999)", "Previous\nFailure", "Previous\nSuccess"])
ax1.set_ylabel("Subscription Rate (%)")
ax1.set_title("A. Conversion by Prior Outcome & Contact Channel")
ax1.set_ylim(0, max(piv.max()) * 1.22)
ax1.legend(frameon=True, facecolor=LIGHT_BG, edgecolor="none")
ax1.grid(axis="y")
for bars in (b_cell, b_tel):
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY)

# Panel B: pdays recency distribution for previously contacted clients (pdays < 999)
prev_df = df[df["pdays"] < 999].copy()
prev_df["pdays_bucket"] = pd.cut(prev_df["pdays"], bins=[0, 4, 8, 14, 30], labels=["1-4 Days", "5-8 Days", "9-14 Days", "15-27 Days"])
grp_pdays = prev_df.groupby("pdays_bucket", observed=False)["y"].agg(["mean", "count"]).reset_index()
bars_p = ax2.bar(grp_pdays["pdays_bucket"].astype(str), grp_pdays["mean"] * 100, color=TEAL, width=0.55)
ax2.set_title("B. Conversion by Days Since Prior Contact (pdays < 999)")
ax2.set_ylabel("Subscription Rate (%)")
ax2.set_ylim(0, max(grp_pdays["mean"] * 100) * 1.25)
ax2.grid(axis="y")
for b, cnt in zip(bars_p, grp_pdays["count"]):
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=9, fontweight="bold", color=NAVY)

fig.tight_layout(pad=2.0)
save_both(fig, "fig2_poutcome_contact.png")

# Figure 3: Campaign Contact Fatigue & Call Duration Target Leakage (T4 Justification)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
df["camp_cap"] = df["campaign"].clip(upper=8).fillna(1).astype(int)
camp_grp = df.groupby("camp_cap")["y"].agg(["mean", "count"]).reset_index()
ax1.plot(camp_grp["camp_cap"], camp_grp["mean"] * 100, marker="o", linewidth=2.5, markersize=7, color=BLUE)
ax1.fill_between(camp_grp["camp_cap"], 0, camp_grp["mean"] * 100, color=BLUE, alpha=0.12)
ax1.set_xticks(range(1, 9))
ax1.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8+"])
ax1.set_xlabel("Number of Contacts in Current Campaign (campaign)")
ax1.set_ylabel("Subscription Rate (%)")
ax1.set_title("A. Diminishing Returns & Contact Fatigue")
ax1.set_ylim(0, max(camp_grp["mean"] * 100) * 1.25)
ax1.grid(True)
for x_v, y_v in zip(camp_grp["camp_cap"], camp_grp["mean"] * 100):
    ax1.text(x_v, y_v + 0.6, f"{y_v:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=NAVY)

# Panel B: Call Duration distribution by target y (illustrating why duration is target leakage)
dur_no = df.loc[df["y"] == 0, "duration"].clip(upper=1500)
dur_yes = df.loc[df["y"] == 1, "duration"].clip(upper=1500)
bins = np.linspace(0, 1500, 31)
ax2.hist(dur_no, bins=bins, density=True, alpha=0.55, color=SLATE, label=f"No Subscription (Median: {df.loc[df['y']==0, 'duration'].median():.0f}s)")
ax2.hist(dur_yes, bins=bins, density=True, alpha=0.60, color=CORAL, label=f"Subscribed (Median: {df.loc[df['y']==1, 'duration'].median():.0f}s)")
ax2.set_xlabel("Last Call Duration in Seconds (Clipped at 1,500s)")
ax2.set_ylabel("Density")
ax2.set_title("B. Post-Call Duration Leakage (Excluded per Check T4)")
ax2.legend(frameon=True, facecolor=LIGHT_BG, edgecolor="none", fontsize=9)
ax2.grid(True)

fig.tight_layout(pad=2.0)
save_both(fig, "fig3_campaign_fatigue.png")

# Figure 4: Model Evaluation on Test Set with 95% Bootstrap CIs (T1, T2, T6)
fig, ax = plt.subplots(figsize=(8.8, 4.2))
order = ["majority_class", "spark_logistic", "sklearn_boosting"]
labels = [
    "Majority Class Baseline\n(family: baseline | engine: other)",
    "Spark MLlib Logistic Regression\n(family: linear | engine: spark)",
    "Scikit-Learn HistGradientBoosting\n(family: tree ensemble | engine: sklearn)"
]
met_sorted = met.set_index("model").loc[order].reset_index()
vals = met_sorted["value"].to_numpy()
lo = met_sorted["ci_low"].to_numpy()
hi = met_sorted["ci_high"].to_numpy()
y_pos = np.arange(len(order))
colors = [SLATE, BLUE, TEAL]

bars = ax.barh(y_pos, vals, height=0.52, color=colors, alpha=0.9)
ax.errorbar(vals, y_pos, xerr=[vals - lo, hi - vals], fmt="none", ecolor=NAVY, elinewidth=2.2, capsize=6, capthick=2.0)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
ax.set_xlabel("Test-Set Macro F1 Score (f1_macro) with 95% Bootstrap CI (1,000 Resamples)")
ax.set_title("Test-Set Primary Metric (f1_macro) & Non-Overlapping 95% Bootstrap Intervals")
ax.set_xlim(0, max(hi) * 1.22)
ax.grid(axis="x")

for i, (v, l, h) in enumerate(zip(vals, lo, hi)):
    ax.text(h + 0.012, y_pos[i], f"{v:.4f}  [95% CI: {l:.4f}, {h:.4f}]", va="center", ha="left", fontsize=9.8, fontweight="bold", color=NAVY)

fig.tight_layout()
save_both(fig, "fig4_model_comparison.png")
print("Successfully generated all 4 EDA and model comparison figures.")
