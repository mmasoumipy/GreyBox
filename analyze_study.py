import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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
    """Extract and clean survey response data."""
    surveys = df[df['event'] == 'survey_completed'].copy()
    
    if surveys.empty:
        print("❌ No survey responses found!")
        return None
    
    print(f"✅ Found {len(surveys)} survey responses")
    
    # Count by group
    group_counts = surveys['group'].value_counts()
    print(f"\nGroup Distribution:")
    for group, count in group_counts.items():
        print(f"  {group}: {count} participants")
    
    return surveys

def compute_statistics(surveys):
    """Compute descriptive and inferential statistics."""
    
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    metrics = ['trust_score', 'follow_likelihood', 'usefulness']
    metric_names = ['Trust in AI', 'Likelihood to Follow', 'Perceived Usefulness']
    
    results = []
    
    for metric, name in zip(metrics, metric_names):
        print(f"\n📊 {name}")
        print("-" * 40)
        
        g1_data = surveys[surveys['group'] == 'G1'][metric].dropna()
        g2_data = surveys[surveys['group'] == 'G2'][metric].dropna()
        
        if len(g1_data) == 0 or len(g2_data) == 0:
            print(f"  ⚠️  Insufficient data for {metric}")
            continue
        
        # Descriptive statistics
        g1_mean = g1_data.mean()
        g1_std = g1_data.std()
        g2_mean = g2_data.mean()
        g2_std = g2_data.std()
        
        print(f"  G1 (Basic):    {g1_mean:.2f} ± {g1_std:.2f} (n={len(g1_data)})")
        print(f"  G2 (Enhanced): {g2_mean:.2f} ± {g2_std:.2f} (n={len(g2_data)})")
        
        # T-test
        t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(g1_data)-1)*g1_std**2 + (len(g2_data)-1)*g2_std**2) / 
                            (len(g1_data) + len(g2_data) - 2))
        cohen_d = (g2_mean - g1_mean) / pooled_std if pooled_std > 0 else 0
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pval = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"  t-statistic:   {t_stat:.3f}")
        print(f"  p-value:       {p_value:.4f} {significance}")
        print(f"  Cohen's d:     {cohen_d:.3f}", end="")
        
        # Effect size interpretation
        if abs(cohen_d) < 0.2:
            print(" (negligible)")
        elif abs(cohen_d) < 0.5:
            print(" (small)")
        elif abs(cohen_d) < 0.8:
            print(" (medium)")
        else:
            print(" (large)")
        
        print(f"  Mann-Whitney U p-value: {u_pval:.4f}")
        
        results.append({
            'Metric': name,
            'G1_Mean': g1_mean,
            'G1_SD': g1_std,
            'G2_Mean': g2_mean,
            'G2_SD': g2_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'Cohen_d': cohen_d,
            'significance': significance
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return results_df

def analyze_qualitative(surveys):
    """Analyze open-ended comments."""
    
    print("\n" + "="*60)
    print("QUALITATIVE ANALYSIS")
    print("="*60)
    
    g1_comments = surveys[surveys['group'] == 'G1']['comments'].dropna()
    g2_comments = surveys[surveys['group'] == 'G2']['comments'].dropna()
    
    print(f"\n📝 G1 (Basic) Comments ({len(g1_comments)} responses):")
    print("-" * 40)
    if len(g1_comments) > 0:
        for i, comment in enumerate(g1_comments, 1):
            if comment.strip():
                print(f"{i}. {comment}")
    else:
        print("  (No comments)")
    
    print(f"\n📝 G2 (Enhanced) Comments ({len(g2_comments)} responses):")
    print("-" * 40)
    if len(g2_comments) > 0:
        for i, comment in enumerate(g2_comments, 1):
            if comment.strip():
                print(f"{i}. {comment}")
    else:
        print("  (No comments)")

def create_visualizations(surveys):
    """Create publication-quality visualizations."""
    
    print("\n📊 Creating visualizations...")
    
    metrics = ['trust_score', 'follow_likelihood', 'usefulness']
    metric_names = ['Trust in AI\n(1-5 scale)', 
                    'Likelihood to\nFollow Plan\n(1-5 scale)', 
                    'Perceived\nUsefulness\n(1-5 scale)']
    
    # Figure 1: Box plots comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric, title in zip(axes, metrics, metric_names):
        g1_data = surveys[surveys['group'] == 'G1'][metric].dropna()
        g2_data = surveys[surveys['group'] == 'G2'][metric].dropna()
        
        if len(g1_data) == 0 or len(g2_data) == 0:
            continue
        
        # Create box plot
        box_data = [g1_data, g2_data]
        bp = ax.boxplot(box_data, labels=['G1:\nBasic', 'G2:\nEnhanced'],
                       patch_artist=True, widths=0.6)
        
        # Color the boxes
        colors = ['#3498db', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, data in enumerate(box_data, 1):
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=30, color='black')
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 6)
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean markers
        means = [g1_data.mean(), g2_data.mean()]
        ax.plot([1, 2], means, 'r--', linewidth=2, marker='D', 
                markersize=8, label='Mean')
        
        # Add significance stars if p < 0.05
        t_stat, p_value = stats.ttest_ind(g1_data, g2_data)
        if p_value < 0.05:
            y_max = max(g1_data.max(), g2_data.max())
            y_pos = y_max + 0.3
            ax.plot([1, 2], [y_pos, y_pos], 'k-', linewidth=1.5)
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
            ax.text(1.5, y_pos + 0.1, sig_text, ha='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('results_boxplots.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: results_boxplots.png")
    
    # Figure 2: Bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    g1_means = [surveys[surveys['group'] == 'G1'][m].mean() for m in metrics]
    g2_means = [surveys[surveys['group'] == 'G2'][m].mean() for m in metrics]
    g1_sems = [surveys[surveys['group'] == 'G1'][m].sem() for m in metrics]
    g2_sems = [surveys[surveys['group'] == 'G2'][m].sem() for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, g1_means, width, yerr=g1_sems, 
                   label='G1: Basic', color='#3498db', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, g2_means, width, yerr=g2_sems,
                   label='G2: Enhanced', color='#2ecc71', alpha=0.8, capsize=5)
    
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Comparison of Mean Scores by Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Trust', 'Follow Likelihood', 'Usefulness'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results_barplot.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: results_barplot.png")
    
    # Figure 3: Distribution histograms
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, (metric, title) in enumerate(zip(metrics, ['Trust', 'Follow Likelihood', 'Usefulness'])):
        g1_data = surveys[surveys['group'] == 'G1'][metric].dropna()
        g2_data = surveys[surveys['group'] == 'G2'][metric].dropna()
        
        # G1 histogram
        axes[i, 0].hist(g1_data, bins=np.arange(0.5, 6.5, 1), 
                       color='#3498db', alpha=0.7, edgecolor='black')
        axes[i, 0].set_title(f'G1: Basic - {title}', fontsize=11, fontweight='bold')
        axes[i, 0].set_xlabel('Score', fontsize=10)
        axes[i, 0].set_ylabel('Frequency', fontsize=10)
        axes[i, 0].set_xlim(0, 6)
        axes[i, 0].grid(axis='y', alpha=0.3)
        
        # G2 histogram
        axes[i, 1].hist(g2_data, bins=np.arange(0.5, 6.5, 1),
                       color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[i, 1].set_title(f'G2: Enhanced - {title}', fontsize=11, fontweight='bold')
        axes[i, 1].set_xlabel('Score', fontsize=10)
        axes[i, 1].set_ylabel('Frequency', fontsize=10)
        axes[i, 1].set_xlim(0, 6)
        axes[i, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_distributions.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: results_distributions.png")
    
    plt.close('all')

def analyze_correlations(surveys):
    """Analyze correlations between metrics."""
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    metrics = ['trust_score', 'follow_likelihood', 'usefulness']
    
    for group in ['G1', 'G2']:
        print(f"\n{group} Correlations:")
        print("-" * 40)
        
        group_data = surveys[surveys['group'] == group][metrics].dropna()
        
        if len(group_data) < 3:
            print("  Insufficient data")
            continue
        
        corr_matrix = group_data.corr()
        print(corr_matrix.round(3))
        
        # Test significance
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
    surveys.to_csv('survey_responses.csv', index=False)
    print("  ✅ Saved: survey_responses.csv")
    
    # Export statistical results
    results_df.to_csv('statistical_results.csv', index=False)
    print("  ✅ Saved: statistical_results.csv")
    
    # Export summary statistics
    summary = surveys.groupby('group')[['trust_score', 'follow_likelihood', 'usefulness']].agg(['mean', 'std', 'count'])
    summary.to_csv('summary_statistics.csv')
    print("  ✅ Saved: summary_statistics.csv")

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
        report.append(f"  G1 (Basic):    M = {row['G1_Mean']:.2f}, SD = {row['G1_SD']:.2f}")
        report.append(f"  G2 (Enhanced): M = {row['G2_Mean']:.2f}, SD = {row['G2_SD']:.2f}")
        report.append(f"  t({len(surveys)-2}) = {row['t_statistic']:.3f}, p = {row['p_value']:.4f} {row['significance']}")
        report.append(f"  Cohen's d = {row['Cohen_d']:.3f}")
        
        # Interpretation
        if row['p_value'] < 0.05:
            direction = "higher" if row['G2_Mean'] > row['G1_Mean'] else "lower"
            report.append(f"  → G2 showed significantly {direction} scores than G1")
        else:
            report.append(f"  → No significant difference between groups")
    
    report.append("")
    report.append("="*70)
    report.append("INTERPRETATION")
    report.append("="*70)
    
    # Calculate overall effect
    trust_diff = results_df[results_df['Metric'] == 'Trust in AI']['G2_Mean'].values[0] - \
                 results_df[results_df['Metric'] == 'Trust in AI']['G1_Mean'].values[0]
    
    if trust_diff > 0:
        report.append("\nShowing uncertainty and AI explanations appears to INCREASE user trust.")
        report.append("This suggests that transparency may be beneficial for AI adoption.")
    else:
        report.append("\nShowing uncertainty and AI explanations appears to DECREASE user trust.")
        report.append("This suggests users may prefer confident predictions over transparent uncertainty.")
    
    report.append("")
    report.append("Note: These are preliminary interpretations. Consider context and limitations.")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    with open('study_report.txt', 'w') as f:
        f.write(report_text)
    
    print("  ✅ Saved: study_report.txt")
    
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
    results_df = compute_statistics(surveys)
    analyze_qualitative(surveys)
    analyze_correlations(surveys)
    create_visualizations(surveys)
    export_results(surveys, results_df)
    generate_report(surveys, results_df)
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 results_boxplots.png")
    print("  📊 results_barplot.png")
    print("  📊 results_distributions.png")
    print("  📄 survey_responses.csv")
    print("  📄 statistical_results.csv")
    print("  📄 summary_statistics.csv")
    print("  📄 study_report.txt")
    print("\n")

if __name__ == "__main__":
    main()