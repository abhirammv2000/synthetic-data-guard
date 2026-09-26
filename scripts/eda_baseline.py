import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────
DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "reports/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# consistent styling
sns.set_theme(style="whitegrid", palette="muted")
FRAUD_COLORS = {0: "#2ecc71", 1: "#e74c3c"}  # green=legit, red=fraud
FIG_DPI = 150


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ─── LOAD DATA ────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes.value_counts().to_string()}")
print(f"\nNull values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")


# ─── 1. CLASS DISTRIBUTION ───────────────────────────────────────────
print("1. CLASS DISTRIBUTION (Fraud vs Legitimate)")


class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print(f"Legitimate (0): {class_counts[0]:>7,}  ({class_pct[0]:.3f}%)")
print(f"Fraud (1):      {class_counts[1]:>7,}  ({class_pct[1]:.3f}%)")
print(f"Imbalance ratio: 1:{class_counts[0] // class_counts[1]}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Class Distribution: Fraud vs Legitimate", fontsize=14, fontweight="bold")

# bar chart
bars = axes[0].bar(["Legitimate", "Fraud"], class_counts.values,
                    color=[FRAUD_COLORS[0], FRAUD_COLORS[1]], edgecolor="white", linewidth=1.5)
axes[0].set_ylabel("Count")
axes[0].set_title("Absolute Counts")
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                 f"{count:,}", ha="center", va="bottom", fontweight="bold")

# log-scale bar for better visibility
axes[1].bar(["Legitimate", "Fraud"], class_counts.values,
            color=[FRAUD_COLORS[0], FRAUD_COLORS[1]], edgecolor="white", linewidth=1.5)
axes[1].set_yscale("log")
axes[1].set_ylabel("Count (log scale)")
axes[1].set_title("Log Scale, See the Fraud")

save_fig(fig, "01_class_distribution.png")


# ─── 2. AMOUNT DISTRIBUTION ──────────────────────────────────────────

print("2. AMOUNT DISTRIBUTION")


print(f"\nOverall Amount stats:")
print(df["Amount"].describe().to_string())
print(f"\nFraud Amount stats:")
print(df[df["Class"] == 1]["Amount"].describe().to_string())
print(f"\nLegitimate Amount stats:")
print(df[df["Class"] == 0]["Amount"].describe().to_string())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Transaction Amount Analysis", fontsize=14, fontweight="bold")

# overall distribution
axes[0, 0].hist(df["Amount"], bins=100, color="#3498db", edgecolor="white", alpha=0.8)
axes[0, 0].set_title("Amount Distribution (All)")
axes[0, 0].set_xlabel("Amount ($)")
axes[0, 0].set_ylabel("Frequency")

# log-transformed amount
log_amount = np.log1p(df["Amount"])
axes[0, 1].hist(log_amount, bins=100, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[0, 1].set_title("log(1 + Amount) Distribution")
axes[0, 1].set_xlabel("log(1 + Amount)")

# fraud vs legit, box plot
data_for_box = [df[df["Class"] == 0]["Amount"], df[df["Class"] == 1]["Amount"]]
bp = axes[1, 0].boxplot(data_for_box, labels=["Legitimate", "Fraud"], patch_artist=True,
                         showfliers=True, flierprops=dict(markersize=2, alpha=0.3))
bp["boxes"][0].set_facecolor(FRAUD_COLORS[0])
bp["boxes"][1].set_facecolor(FRAUD_COLORS[1])
axes[1, 0].set_title("Amount by Class")
axes[1, 0].set_ylabel("Amount ($)")

# fraud vs legit, overlapping histograms (zoomed to <2000)
mask_legit = (df["Class"] == 0) & (df["Amount"] < 2000)
mask_fraud = (df["Class"] == 1) & (df["Amount"] < 2000)
axes[1, 1].hist(df.loc[mask_legit, "Amount"], bins=80, alpha=0.6,
                color=FRAUD_COLORS[0], label="Legitimate", density=True)
axes[1, 1].hist(df.loc[mask_fraud, "Amount"], bins=80, alpha=0.6,
                color=FRAUD_COLORS[1], label="Fraud", density=True)
axes[1, 1].set_title("Amount < $2000 (Density, by Class)")
axes[1, 1].set_xlabel("Amount ($)")
axes[1, 1].legend()

plt.tight_layout()
save_fig(fig, "02_amount_distribution.png")


# ─── 3. TIME DISTRIBUTION ────────────────────────────────────────────

print("3. TIME DISTRIBUTION")


time_hours = df["Time"] / 3600
print(f"Time range: {time_hours.min():.1f} to {time_hours.max():.1f} hours")
print(f"  → Covers ~{time_hours.max() / 24:.1f} days of transactions")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Transaction Time Analysis", fontsize=14, fontweight="bold")

# overall time distribution
axes[0].hist(time_hours, bins=48, color="#3498db", edgecolor="white", alpha=0.8)
axes[0].set_xlabel("Time (hours from start)")
axes[0].set_ylabel("Transaction Count")
axes[0].set_title("All Transactions Over Time")

# fraud vs legit over time
axes[1].hist(time_hours[df["Class"] == 0], bins=48, alpha=0.5,
             color=FRAUD_COLORS[0], label="Legitimate", density=True)
axes[1].hist(time_hours[df["Class"] == 1], bins=48, alpha=0.5,
             color=FRAUD_COLORS[1], label="Fraud", density=True)
axes[1].set_xlabel("Time (hours from start)")
axes[1].set_ylabel("Density")
axes[1].set_title("Fraud vs Legitimate Over Time")
axes[1].legend()

plt.tight_layout()
save_fig(fig, "03_time_distribution.png")


# ─── 4. V-FEATURE DISTRIBUTIONS ──────────────────────────────────────

print("4. PCA FEATURE DISTRIBUTIONS (V1-V28)")


v_features = [f"V{i}" for i in range(1, 29)]
v_stats = df[v_features].describe().T
print(f"\nV-feature summary stats (transposed):")
print(v_stats[["mean", "std", "min", "max"]].to_string())

# which V-features differ most between fraud and legit?
v_mean_diff = {}
for col in v_features:
    legit_mean = df[df["Class"] == 0][col].mean()
    fraud_mean = df[df["Class"] == 1][col].mean()
    v_mean_diff[col] = abs(fraud_mean - legit_mean)

v_diff_sorted = sorted(v_mean_diff.items(), key=lambda x: x[1], reverse=True)
print("\nV-features with LARGEST fraud vs legit mean difference:")
for feat, diff in v_diff_sorted[:10]:
    f_mean = df[df["Class"] == 1][feat].mean()
    l_mean = df[df["Class"] == 0][feat].mean()
    print(f"  {feat:>4}: legit={l_mean:>8.4f}  fraud={f_mean:>8.4f}  |diff|={diff:.4f}")

# plot top 8 most discriminative V-features
top_v = [x[0] for x in v_diff_sorted[:8]]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
fig.suptitle("Top 8 Most Discriminative V-Features (Fraud vs Legitimate)", fontsize=14, fontweight="bold")

for idx, feat in enumerate(top_v):
    ax = axes[idx // 4, idx % 4]
    ax.hist(df.loc[df["Class"] == 0, feat], bins=60, alpha=0.5,
            color=FRAUD_COLORS[0], label="Legit", density=True)
    ax.hist(df.loc[df["Class"] == 1, feat], bins=60, alpha=0.5,
            color=FRAUD_COLORS[1], label="Fraud", density=True)
    ax.set_title(feat, fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
save_fig(fig, "04_top_v_features.png")

# all 28 V-feature distributions, small multiples
fig, axes = plt.subplots(4, 7, figsize=(24, 12))
fig.suptitle("All V-Feature Distributions (V1-V28)", fontsize=14, fontweight="bold")

for idx, feat in enumerate(v_features):
    ax = axes[idx // 7, idx % 7]
    ax.hist(df[feat], bins=50, color="#3498db", edgecolor="none", alpha=0.7)
    ax.set_title(feat, fontsize=9)
    ax.tick_params(labelsize=7)

plt.tight_layout()
save_fig(fig, "05_all_v_features.png")


# ─── 5. CORRELATION ANALYSIS ─────────────────────────────────────────

print("5. CORRELATION ANALYSIS")


# full correlation matrix
corr_matrix = df.corr()

# correlation with Class (fraud indicator)
class_corr = corr_matrix["Class"].drop("Class").sort_values()
print("\nTop features correlated with FRAUD (Class):")
print("  Most NEGATIVE (fraud indicator):")
for feat in class_corr.head(5).index:
    print(f"    {feat:>8}: {class_corr[feat]:>+.4f}")
print("  Most POSITIVE (fraud indicator):")
for feat in class_corr.tail(5).index:
    print(f"    {feat:>8}: {class_corr[feat]:>+.4f}")

# V-feature correlation heatmap
fig, ax = plt.subplots(figsize=(16, 13))
v_corr = df[v_features].corr()
mask = np.triu(np.ones_like(v_corr, dtype=bool), k=1)
sns.heatmap(v_corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=False, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
ax.set_title("V-Feature Correlation Matrix (Lower Triangle)", fontsize=14, fontweight="bold")
save_fig(fig, "06_v_feature_correlations.png")

# correlation with Class, bar chart
fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in class_corr.values]
class_corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
ax.set_title("Feature Correlation with Fraud (Class)", fontsize=14, fontweight="bold")
ax.set_xlabel("Pearson Correlation")
ax.axvline(x=0, color="black", linewidth=0.5)
save_fig(fig, "07_class_correlations.png")


# ─── 6. OUTLIER DETECTION ────────────────────────────────────────────

print("6. OUTLIER DETECTION")


# IQR method on Amount
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
amount_outliers = df[(df["Amount"] < lower) | (df["Amount"] > upper)]
print(f"\nAmount outliers (IQR method):")
print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
print(f"  Bounds: [{lower:.2f}, {upper:.2f}]")
print(f"  Outliers: {len(amount_outliers):,} ({len(amount_outliers)/len(df)*100:.2f}%)")
print(f"  Outliers that are fraud: {amount_outliers['Class'].sum()} / {len(amount_outliers)}")

# Z-score outliers across V-features
z_scores = np.abs(stats.zscore(df[v_features], nan_policy="omit"))
extreme_outliers = (z_scores > 5).sum(axis=0)  # z > 5 is very extreme
print(f"\nV-feature extreme outliers (|z| > 5):")
outlier_df = pd.DataFrame({"feature": v_features, "count": extreme_outliers})
outlier_df = outlier_df.sort_values("count", ascending=False).head(10)
for _, row in outlier_df.iterrows():
    print(f"  {row['feature']:>4}: {row['count']:>5} extreme values")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Outlier Analysis", fontsize=14, fontweight="bold")

# Amount outliers scatter
axes[0].scatter(df.index[df["Amount"] <= upper], df.loc[df["Amount"] <= upper, "Amount"],
                s=1, alpha=0.1, color="#3498db", label="Normal")
axes[0].scatter(amount_outliers.index, amount_outliers["Amount"],
                s=3, alpha=0.3, color="#e74c3c", label="Outlier")
axes[0].set_title("Amount Outliers (IQR Method)")
axes[0].set_xlabel("Transaction Index")
axes[0].set_ylabel("Amount ($)")
axes[0].legend()

# extreme z-scores per V-feature
axes[1].bar(outlier_df["feature"], outlier_df["count"], color="#e74c3c", edgecolor="white")
axes[1].set_title("V-Features with Most Extreme Values (|z| > 5)")
axes[1].set_ylabel("Count of Extreme Values")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
save_fig(fig, "08_outlier_analysis.png")


# ─── 7. FRAUD DEEP DIVE ──────────────────────────────────────────────

print("7. FRAUD PATTERN ANALYSIS")


fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

print(f"\nFraud transaction Amount percentiles:")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  {pct}th: ${fraud['Amount'].quantile(pct/100):.2f}")

print(f"\nLegit transaction Amount percentiles:")
for pct in [25, 50, 75, 90, 95, 99]:
    print(f"  {pct}th: ${legit['Amount'].quantile(pct/100):.2f}")

# KS test: how different are fraud vs legit distributions?
print(f"\nKolmogorov-Smirnov tests (fraud vs legit):")
ks_results = {}
for col in ["Amount", "Time"] + v_features:
    ks_stat, ks_pval = stats.ks_2samp(fraud[col], legit[col])
    ks_results[col] = {"statistic": ks_stat, "p_value": ks_pval}
    if ks_stat > 0.3:  # only print notably different ones
        print(f"  {col:>8}: KS={ks_stat:.4f}  p={ks_pval:.2e}  {'***' if ks_pval < 0.001 else ''}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Fraud Pattern Deep Dive", fontsize=14, fontweight="bold")

# fraud amount distribution
axes[0, 0].hist(fraud["Amount"], bins=50, color=FRAUD_COLORS[1], edgecolor="white", alpha=0.8)
axes[0, 0].set_title("Fraud Amount Distribution")
axes[0, 0].set_xlabel("Amount ($)")

# fraud timing
fraud_hours = fraud["Time"] / 3600
axes[0, 1].hist(fraud_hours, bins=24, color=FRAUD_COLORS[1], edgecolor="white", alpha=0.8)
axes[0, 1].set_title("Fraud Transactions Over Time")
axes[0, 1].set_xlabel("Hours from Start")

# fraud amount CDF vs legit
sorted_legit = np.sort(legit["Amount"])
sorted_fraud = np.sort(fraud["Amount"])
axes[0, 2].plot(sorted_legit, np.linspace(0, 1, len(sorted_legit)),
                color=FRAUD_COLORS[0], label="Legitimate", alpha=0.7)
axes[0, 2].plot(sorted_fraud, np.linspace(0, 1, len(sorted_fraud)),
                color=FRAUD_COLORS[1], label="Fraud", alpha=0.7)
axes[0, 2].set_title("Amount CDF: Fraud vs Legitimate")
axes[0, 2].set_xlabel("Amount ($)")
axes[0, 2].set_ylabel("Cumulative Probability")
axes[0, 2].set_xlim(0, 500)
axes[0, 2].legend()

# top KS stats: features that best separate fraud from legit
ks_df = pd.DataFrame(ks_results).T.sort_values("statistic", ascending=False).head(10)
axes[1, 0].barh(ks_df.index, ks_df["statistic"], color="#e74c3c", edgecolor="white")
axes[1, 0].set_title("KS Statistic: Fraud vs Legit (Top 10)")
axes[1, 0].set_xlabel("KS Statistic")
axes[1, 0].invert_yaxis()

# V14 and V17 scatter (typically most discriminative)
top2 = [x[0] for x in v_diff_sorted[:2]]
axes[1, 1].scatter(legit[top2[0]], legit[top2[1]], s=1, alpha=0.05,
                    color=FRAUD_COLORS[0], label="Legit")
axes[1, 1].scatter(fraud[top2[0]], fraud[top2[1]], s=5, alpha=0.3,
                    color=FRAUD_COLORS[1], label="Fraud")
axes[1, 1].set_xlabel(top2[0])
axes[1, 1].set_ylabel(top2[1])
axes[1, 1].set_title(f"Fraud Separation: {top2[0]} vs {top2[1]}")
axes[1, 1].legend()

# PCA-style 2D projection (using top 2 correlated V-features with Class)
most_neg = class_corr.head(1).index[0]
most_pos = class_corr.tail(1).index[0]
axes[1, 2].scatter(legit[most_neg], legit[most_pos], s=1, alpha=0.05,
                    color=FRAUD_COLORS[0], label="Legit")
axes[1, 2].scatter(fraud[most_neg], fraud[most_pos], s=5, alpha=0.3,
                    color=FRAUD_COLORS[1], label="Fraud")
axes[1, 2].set_xlabel(f"{most_neg} (most -corr)")
axes[1, 2].set_ylabel(f"{most_pos} (most +corr)")
axes[1, 2].set_title("Top Fraud-Correlated Features")
axes[1, 2].legend()

plt.tight_layout()
save_fig(fig, "09_fraud_patterns.png")


# ─── 8. BASELINE STATS (for synthetic data comparison) ───────────────

print("8. SAVING BASELINE STATISTICS")

baseline = {
    "metadata": {
        "source": "creditcard.csv",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "fraud_rate": float(class_pct[1] / 100),
        "generated_by": "eda_baseline.py",
    },
    "column_stats": {},
    "correlations": {
        "class_correlations": {k: round(float(v), 6) for k, v in class_corr.items()},
    },
    "ks_baselines": {},
    "distribution_params": {},
}

# per-column stats
for col in df.columns:
    col_data = df[col]
    baseline["column_stats"][col] = {
        "mean": round(float(col_data.mean()), 6),
        "std": round(float(col_data.std()), 6),
        "min": round(float(col_data.min()), 6),
        "max": round(float(col_data.max()), 6),
        "median": round(float(col_data.median()), 6),
        "q25": round(float(col_data.quantile(0.25)), 6),
        "q75": round(float(col_data.quantile(0.75)), 6),
        "skewness": round(float(col_data.skew()), 6),
        "kurtosis": round(float(col_data.kurtosis()), 6),
    }

# save V-feature correlation matrix for later comparison
v_corr_dict = {}
for i, col1 in enumerate(v_features):
    for j, col2 in enumerate(v_features):
        if i < j:
            v_corr_dict[f"{col1}__{col2}"] = round(float(v_corr.loc[col1, col2]), 6)
baseline["correlations"]["v_feature_pairs"] = v_corr_dict

# KS test baselines (fraud vs legit), useful for validating synthetic fraud patterns
for col in ["Amount", "Time"] + v_features:
    ks_stat, ks_pval = stats.ks_2samp(fraud[col], legit[col])
    baseline["ks_baselines"][col] = {
        "statistic": round(float(ks_stat), 6),
        "p_value": float(ks_pval),
    }

# fit basic distribution params for Amount (useful for rule-based validation later)
# Amount follows something like a log-normal
log_amount = np.log1p(df["Amount"])
baseline["distribution_params"]["log_amount"] = {
    "mean": round(float(log_amount.mean()), 6),
    "std": round(float(log_amount.std()), 6),
    "shapiro_stat": round(float(stats.shapiro(log_amount.sample(5000, random_state=42))[0]), 6),
}

baseline_path = os.path.join(OUTPUT_DIR, "baseline_stats.json")
with open(baseline_path, "w") as f:
    json.dump(baseline, f, indent=2)
print(f"  ✓ Saved {baseline_path}")
print(f"    → {len(baseline['column_stats'])} column stat profiles")
print(f"    → {len(baseline['ks_baselines'])} KS test baselines")
print(f"    → {len(baseline['correlations']['v_feature_pairs'])} V-feature correlation pairs")


# ─── 9. SUMMARY ──────────────────────────────────────────────────────

print("EDA SUMMARY: KEY FINDINGS")

print(f"""
Dataset: 284,807 transactions over ~48 hours
  • Only {class_counts[1]} frauds ({class_pct[1]:.3f}%), extreme class imbalance (1:{class_counts[0]//class_counts[1]})
  • V1-V28 are PCA-transformed (already normalized, zero mean expected)
  • Amount is right-skewed (median ${df['Amount'].median():.2f}, max ${df['Amount'].max():.2f})
  • Time shows clear cyclical patterns (day/night transaction volumes)

Fraud characteristics:
  • Fraud amounts are generally LOWER (median ${fraud['Amount'].median():.2f} vs ${legit['Amount'].median():.2f})
  • Most discriminative features: {', '.join([x[0] for x in v_diff_sorted[:5]])}
  • Several V-features show strong fraud separation (useful for validation rules later)

For synthetic data generation (Week 2-3):
  • Must preserve the {class_pct[1]:.3f}% fraud rate
  • Must maintain V-feature correlation structure (they're near-zero since PCA already decorrelated them)
  • Amount distribution shape is critical (log-normal-ish)
  • Time cyclicality should be replicated
  • Baseline stats saved to {baseline_path} for automated comparison

Visualizations saved to {OUTPUT_DIR}/:
""")

for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  • {f}")

print(f"\nTotal output files: {len(os.listdir(OUTPUT_DIR))}")
