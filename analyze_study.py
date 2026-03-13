import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

OUTPUT_DIR = Path("study_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Aggregated metric definitions for the S-TIAS questionnaire
AGG_METRICS = {
    "stias_trust": {
        "label": "S-TIAS - Trust & Reliability",
        "questions": [
            "q_stias_reliable",
            "q_stias_trust",
            "q_stias_understandable",
        ],
    },
    "ai_impact": {
        "label": "AI Impact",
        "questions": [
            "q_ai_reasonable",
            "q_ai_trust_recs",
            "q_ai_understand_factors",
            "q_ai_recs_useful",
        ],
    },
    "intentions": {
        "label": "Intentions (Next 7 Days)",
        "questions": [
            "q_intent_stress_technique",
        ],
    },
    "gaais_attitudes": {
        "label": "GAAIS Attitudes",
        "questions": [
            "q_gaais_interest",
            "q_gaais_impact",
            "q_gaais_excitement",
            "q_gaais_benefit",
            "q_gaais_employment",
            "q_gaais_sinister",
            "q_gaais_control",
            "q_gaais_danger",
            "q_gaais_discomfort",
            "q_gaais_suffering",
        ],
    },
    "g2_confidence_info": {
        "label": "Model Confidence Information (G2)",
        "questions": [
            "q_g2_confidence_certainty",
            "q_g2_confidence_trust",
            "q_g2_transparency",
            "q_g2_confidence_clear",
            "q_g2_understand_types",
        ],
        "g2_only": True,
    },
}

def load_study_data():
    """Load all session logs from study_logs directory."""
    log_dir = Path("study_logs")
    
    if not log_dir.exists():
        print("❌ No study_logs directory found!")
        return None
    
    log_files = list(log_dir.glob("*.json"))
    
    if not log_files:
        print("❌ No log files found in study_logs/")
        return None
    
    print(f"📂 Found {len(log_files)} session files")
    
    all_data = []
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"⚠️  Error loading {log_file}: {e}")
    
    df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(df)} total events")
    
    return df

def extract_survey_data(df):
    """Extract and clean questionnaire responses."""
    surveys = df[df["event"] == "questionnaire_completed"].copy()

    if surveys.empty:
        print("❌ No questionnaire responses found!")
        return None

    print(f"✅ Found {len(surveys)} questionnaire responses")

    # Expand nested structures
    response_df = pd.json_normalize(surveys["responses"]).add_prefix("resp.")
    demo_df = pd.json_normalize(surveys.get("demographics", {})).add_prefix("demo.")

    surveys = surveys.reset_index(drop=True)
    surveys_expanded = pd.concat(
        [
            surveys[["user_id", "group", "timestamp"]],
            response_df,
            demo_df,
        ],
        axis=1,
    )

    # Count by group
    group_counts = surveys_expanded["group"].value_counts()
    print(f"\nGroup Distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} participants")

    return surveys_expanded

def compute_statistics(surveys):
    """Compute descriptive and inferential statistics for aggregated metrics."""

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    results = []

    for key, meta in AGG_METRICS.items():
        label = meta["label"]
        questions = [f"resp.{q}" for q in meta["questions"]]
        g2_only = meta.get("g2_only", False)

        # Build composite scores
        for grp in ["G1", "G2"]:
            mask = surveys["group"] == grp
            grp_scores = surveys.loc[mask, questions].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            surveys.loc[mask, key] = grp_scores

        g1_data = surveys.loc[surveys["group"] == "G1", key].dropna()
        g2_data = surveys.loc[surveys["group"] == "G2", key].dropna()

        if g2_only and g2_data.empty:
            continue
        if len(g1_data) == 0 and len(g2_data) == 0:
            print(f"  ⚠️  No data for {label}")
            continue

        print(f"\n📊 {label}")
        print("-" * 40)

        g1_mean = g1_data.mean() if len(g1_data) else np.nan
        g1_std = g1_data.std() if len(g1_data) else np.nan
        g2_mean = g2_data.mean() if len(g2_data) else np.nan
        g2_std = g2_data.std() if len(g2_data) else np.nan

        if not g2_only:
            print(f"  G1 (Basic):    {g1_mean:.2f} ± {g1_std:.2f} (n={len(g1_data)})")
        print(f"  G2 (Enhanced): {g2_mean:.2f} ± {g2_std:.2f} (n={len(g2_data)})")

        total_n = len(g1_data) + len(g2_data)
        if g2_only or len(g1_data) == 0 or len(g2_data) == 0 or total_n < 3:
            significance = "n/a"
            t_stat = np.nan
            p_value = np.nan
            cohen_d = np.nan
            u_pval = np.nan
        else:
            t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
            pooled_std = np.sqrt(
                ((len(g1_data) - 1) * g1_std**2 + (len(g2_data) - 1) * g2_std**2)
                / max(total_n - 2, 1)
            )
            cohen_d = (g2_mean - g1_mean) / pooled_std if pooled_std > 0 else 0
            _, u_pval = stats.mannwhitneyu(g1_data, g2_data, alternative="two-sided")
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"  t-statistic:   {t_stat:.3f}")
            print(f"  p-value:       {p_value:.4f} {significance}")
            print(f"  Cohen's d:     {cohen_d:.3f}", end="")
            if abs(cohen_d) < 0.2:
                print(" (negligible)")
            elif abs(cohen_d) < 0.5:
                print(" (small)")
            elif abs(cohen_d) < 0.8:
                print(" (medium)")
            else:
                print(" (large)")
            print(f"  Mann-Whitney U p-value: {u_pval:.4f}")

        results.append(
            {
                "Metric": label,
                "Key": key,
                "G1_Mean": g1_mean,
                "G1_SD": g1_std,
                "G2_Mean": g2_mean,
                "G2_SD": g2_std,
                "t_statistic": t_stat,
                "p_value": p_value,
                "Cohen_d": cohen_d,
                "significance": significance,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    if not results_df.empty:
        print(results_df[["Metric", "G1_Mean", "G2_Mean", "p_value", "Cohen_d", "significance"]].to_string(index=False))
    else:
        print("No metrics computed.")

    return results_df, surveys

def analyze_qualitative(surveys):
    """Analyze open-ended comments from the new questionnaire."""

    print("\n" + "=" * 60)
    print("QUALITATIVE ANALYSIS")
    print("=" * 60)

    text_fields = {
        "resp.q_open_most_useful": "Model confidence impact",
        "resp.q_open_unclear": "Unclear/confusing parts",
        "resp.q_open_suggestions": "Clarity/usefulness improvements",
    }

    for field, label in text_fields.items():
        if field not in surveys.columns:
            continue
        print(f"\n📝 {label}")
        print("-" * 40)
        for group in ["G1", "G2"]:
            subset = surveys.loc[(surveys["group"] == group) & surveys[field].notna(), field]
            if subset.empty:
                continue
            print(f"\n{group} ({len(subset)} responses):")
            for i, comment in enumerate(subset, 1):
                if str(comment).strip():
                    print(f"{i}. {comment}")

def create_visualizations(surveys):
    """Create publication-quality visualizations for key aggregated metrics."""

    print("\n📊 Creating visualizations...")

    metrics = ["trust_confidence", "perceived_accuracy", "usefulness", "willingness_decision_support", "behavioral_intent"]
    metric_labels = [AGG_METRICS[m]["label"] for m in metrics]

    # Figure 1: Box plots comparison
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, metric_labels):
        g1_data = surveys[surveys["group"] == "G1"][metric].dropna()
        g2_data = surveys[surveys["group"] == "G2"][metric].dropna()

        if len(g1_data) == 0 or len(g2_data) == 0:
            ax.set_title(f"{title}\n(insufficient data)")
            ax.axis("off")
            continue

        box_data = [g1_data, g2_data]
        bp = ax.boxplot(
            box_data,
            labels=["G1:\nBasic", "G2:\nEnhanced"],
            patch_artist=True,
            widths=0.6,
        )

        colors = ["#3498db", "#2ecc71"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, data in enumerate(box_data, 1):
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=30, color="black")

        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 6)
        ax.grid(axis="y", alpha=0.3)

        means = [g1_data.mean(), g2_data.mean()]
        ax.plot([1, 2], means, "r--", linewidth=2, marker="D", markersize=8, label="Mean")

        t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
        if p_value < 0.05:
            y_max = max(g1_data.max(), g2_data.max())
            y_pos = y_max + 0.3
            ax.plot([1, 2], [y_pos, y_pos], "k-", linewidth=1.5)
            sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            ax.text(1.5, y_pos + 0.1, sig_text, ha="center", fontsize=14)

    plt.tight_layout()
    boxplot_path = OUTPUT_DIR / "results_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Saved: {boxplot_path}")

    # Figure 2: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    g1_means = [surveys[surveys["group"] == "G1"][m].mean() for m in metrics]
    g2_means = [surveys[surveys["group"] == "G2"][m].mean() for m in metrics]
    g1_sems = [surveys[surveys["group"] == "G1"][m].sem() for m in metrics]
    g2_sems = [surveys[surveys["group"] == "G2"][m].sem() for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, g1_means, width, yerr=g1_sems, label="G1: Basic", color="#3498db", alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width / 2, g2_means, width, yerr=g2_sems, label="G2: Enhanced", color="#2ecc71", alpha=0.8, capsize=5)

    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title("Comparison of Mean Scores by Group", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, rotation=20, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    barplot_path = OUTPUT_DIR / "results_barplot.png"
    plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Saved: {barplot_path}")

    # Figure 3: Distribution histograms
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 4 * len(metrics)))

    for i, (metric, title) in enumerate(zip(metrics, metric_labels)):
        g1_data = surveys[surveys["group"] == "G1"][metric].dropna()
        g2_data = surveys[surveys["group"] == "G2"][metric].dropna()

        axes[i, 0].hist(g1_data, bins=np.arange(0.5, 6.5, 1), color="#3498db", alpha=0.7, edgecolor="black")
        axes[i, 0].set_title(f"G1: Basic - {title}", fontsize=11, fontweight="bold")
        axes[i, 0].set_xlabel("Score", fontsize=10)
        axes[i, 0].set_ylabel("Frequency", fontsize=10)
        axes[i, 0].set_xlim(0, 6)
        axes[i, 0].grid(axis="y", alpha=0.3)

        axes[i, 1].hist(g2_data, bins=np.arange(0.5, 6.5, 1), color="#2ecc71", alpha=0.7, edgecolor="black")
        axes[i, 1].set_title(f"G2: Enhanced - {title}", fontsize=11, fontweight="bold")
        axes[i, 1].set_xlabel("Score", fontsize=10)
        axes[i, 1].set_ylabel("Frequency", fontsize=10)
        axes[i, 1].set_xlim(0, 6)
        axes[i, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    dist_path = OUTPUT_DIR / "results_distributions.png"
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    print(f"  ✅ Saved: {dist_path}")

    plt.close("all")

def analyze_correlations(surveys):
    """Analyze correlations between core aggregated metrics."""

    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    metrics = ["trust_confidence", "perceived_accuracy", "usefulness", "willingness_decision_support", "behavioral_intent"]

    for group in ["G1", "G2"]:
        print(f"\n{group} Correlations:")
        print("-" * 40)

        group_data = surveys[surveys["group"] == group][metrics].dropna()

        if len(group_data) < 3:
            print("  Insufficient data")
            continue

        corr_matrix = group_data.corr()
        print(corr_matrix.round(3))

        print("\n  Correlation p-values:")
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if i < j:
                    r, p = stats.pearsonr(group_data[m1], group_data[m2])
                    sig = "*" if p < 0.05 else ""
                    print(f"    {m1} vs {m2}: r={r:.3f}, p={p:.3f}{sig}")

def export_results(surveys, results_df):
    """Export results to CSV files."""
    
    print("\n📥 Exporting results...")
    
    # Export survey data
    survey_path = OUTPUT_DIR / 'survey_responses.csv'
    surveys.to_csv(survey_path, index=False)
    print(f"  ✅ Saved: {survey_path}")
    
    # Export statistical results
    stats_path = OUTPUT_DIR / 'statistical_results.csv'
    results_df.to_csv(stats_path, index=False)
    print(f"  ✅ Saved: {stats_path}")
    
    # Export summary statistics
    summary_cols = ["trust_confidence", "perceived_accuracy", "usefulness", "willingness", "decision_support"]
    available_cols = [c for c in summary_cols if c in surveys.columns]
    summary = surveys.groupby('group')[available_cols].agg(['mean', 'std', 'count'])
    summary_path = OUTPUT_DIR / 'summary_statistics.csv'
    summary.to_csv(summary_path)
    print(f"  ✅ Saved: {summary_path}")

def generate_report(surveys, results_df):
    """Generate a text report."""
    
    print("\n📄 Generating report...")
    
    report = []
    report.append("="*70)
    report.append("AI TRUST USER STUDY - RESULTS REPORT")
    report.append("="*70)
    report.append("")
    
    # Sample info
    report.append("SAMPLE INFORMATION")
    report.append("-"*70)
    report.append(f"Total participants: {len(surveys)}")
    group_counts = surveys['group'].value_counts()
    for group, count in group_counts.items():
        report.append(f"  {group}: {count} participants")
    report.append("")
    
    # Main findings
    report.append("MAIN FINDINGS")
    report.append("-"*70)
    
    for _, row in results_df.iterrows():
        report.append(f"\n{row['Metric']}:")
        if not np.isnan(row['G1_Mean']):
            report.append(f"  G1 (Basic):    M = {row['G1_Mean']:.2f}, SD = {row['G1_SD']:.2f}")
        report.append(f"  G2 (Enhanced): M = {row['G2_Mean']:.2f}, SD = {row['G2_SD']:.2f}")
        if not np.isnan(row['t_statistic']):
            report.append(f"  t({len(surveys)-2}) = {row['t_statistic']:.3f}, p = {row['p_value']:.4f} {row['significance']}")
            report.append(f"  Cohen's d = {row['Cohen_d']:.3f}")
            if row['p_value'] < 0.05:
                direction = "higher" if row['G2_Mean'] > row['G1_Mean'] else "lower"
                report.append(f"  → G2 showed significantly {direction} scores than G1")
            else:
                report.append(f"  → No significant difference between groups")
        else:
            report.append("  (G2-only metric; no between-group test)")
    
    report.append("")
    report.append("="*70)
    report.append("INTERPRETATION")
    report.append("="*70)
    
    key_metric = results_df[results_df["Key"] == "trust_confidence"]
    if not key_metric.empty and not np.isnan(key_metric.iloc[0]["t_statistic"]):
        trust_diff = key_metric.iloc[0]["G2_Mean"] - key_metric.iloc[0]["G1_Mean"]
        if trust_diff > 0:
            report.append("\nShowing uncertainty and AI explanations appears to INCREASE trust & confidence.")
            report.append("This suggests that transparency may be beneficial for adoption.")
        elif trust_diff < 0:
            report.append("\nShowing uncertainty and AI explanations appears to DECREASE trust & confidence.")
            report.append("Participants may prefer confident predictions over transparent ranges.")
        else:
            report.append("\nNo observed difference in trust between groups.")
    else:
        report.append("\nInsufficient data to compare trust across groups.")
    
    report.append("")
    report.append("Note: These are preliminary interpretations. Consider context and limitations.")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    report_path = OUTPUT_DIR / 'study_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"  ✅ Saved: {report_path}")
    
    # Print to console
    print("\n" + report_text)

def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*70)
    print("AI TRUST USER STUDY - AUTOMATED ANALYSIS")
    print("="*70 + "\n")
    
    # Load data
    df = load_study_data()
    if df is None:
        return
    
    # Extract surveys
    surveys = extract_survey_data(df)
    if surveys is None:
        return
    
    # Check minimum sample size
    group_counts = surveys['group'].value_counts()
    if any(count < 5 for count in group_counts):
        print("\n⚠️  WARNING: Some groups have fewer than 5 participants.")
        print("   Results may not be reliable. Consider collecting more data.")
        proceed = input("\nContinue with analysis anyway? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Run analyses
    results_df, surveys_with_metrics = compute_statistics(surveys)
    analyze_qualitative(surveys_with_metrics)
    analyze_correlations(surveys_with_metrics)
    create_visualizations(surveys_with_metrics)
    export_results(surveys_with_metrics, results_df)
    generate_report(surveys_with_metrics, results_df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print(f"  📊 {OUTPUT_DIR / 'results_boxplots.png'}")
    print(f"  📊 {OUTPUT_DIR / 'results_barplot.png'}")
    print(f"  📊 {OUTPUT_DIR / 'results_distributions.png'}")
    print(f"  📄 {OUTPUT_DIR / 'survey_responses.csv'}")
    print(f"  📄 {OUTPUT_DIR / 'statistical_results.csv'}")
    print(f"  📄 {OUTPUT_DIR / 'summary_statistics.csv'}")
    print(f"  📄 {OUTPUT_DIR / 'study_report.txt'}")
    print("\n")

if __name__ == "__main__":
    main()
