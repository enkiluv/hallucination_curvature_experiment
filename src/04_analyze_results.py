"""
Results Analysis and Visualization - Defensive Version
"""
import json
import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def safe_import_matplotlib():
    """Safe matplotlib import"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[ERROR] Cannot import matplotlib.")
        return None

def safe_import_seaborn():
    """Safe seaborn import"""
    try:
        import seaborn as sns
        return sns
    except ImportError:
        print("[ERROR] Cannot import seaborn.")
        return None

def safe_import_scipy():
    """Safe scipy import"""
    try:
        from scipy import stats
        return stats
    except ImportError:
        print("[ERROR] Cannot import scipy.")
        return None

# Import dependencies
plt = safe_import_matplotlib()
sns = safe_import_seaborn()
stats = safe_import_scipy()

if plt is None or stats is None:
    print("[WARNING] Some visualization features will be limited.")

# Font settings
if plt is not None:
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

def load_results():
    """Load results"""
    results_path = 'results/03_final_results.json'
    if not os.path.exists(results_path):
        print(f"[ERROR] File not found: {results_path}")
        print("   Run 03_compute_h.py first.")
        sys.exit(1)
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return pd.DataFrame(results)
    except Exception as e:
        print(f"[ERROR] Failed to load results: {e}")
        sys.exit(1)

def create_summary_table(df):
    """Create summary table"""
    summary = df[['id', 'category', 'rho', 'kappa', 'kappa_NPR', 'kappa_LLD', 'h']].copy()
    
    # Add predictions
    test_data_path = 'data/test_inputs.json'
    if os.path.exists(test_data_path):
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            predictions = {case['id']: {
                'pred_rho': case.get('predicted_rho', 0),
                'pred_kappa': case.get('predicted_kappa', 0),
                'pred_h': case.get('predicted_h', 0)
            } for case in test_data['test_cases']}
            
            summary['pred_rho'] = summary['id'].map(lambda x: predictions.get(x, {}).get('pred_rho', 0))
            summary['pred_kappa'] = summary['id'].map(lambda x: predictions.get(x, {}).get('pred_kappa', 0))
            summary['pred_h'] = summary['id'].map(lambda x: predictions.get(x, {}).get('pred_h', 0))
        except Exception as e:
            print(f"[WARNING] Could not load predictions: {e}")
    
    return summary

def compute_statistics(df):
    """Statistical analysis"""
    stats_results = {}
    
    if stats is None:
        print("[WARNING] scipy not available, skipping statistical tests.")
        return stats_results
    
    try:
        # kappa-h correlation
        r_kappa_h, p_kappa_h = stats.pearsonr(df['kappa'], df['h'])
        stats_results['kappa_h_correlation'] = {
            'r': float(r_kappa_h),
            'p': float(p_kappa_h),
            'interpretation': 'Strong' if abs(r_kappa_h) > 0.7 else 'Moderate' if abs(r_kappa_h) > 0.4 else 'Weak'
        }
        
        # rho-kappa correlation
        r_rho_kappa, p_rho_kappa = stats.pearsonr(df['rho'], df['kappa'])
        stats_results['rho_kappa_correlation'] = {
            'r': float(r_rho_kappa),
            'p': float(p_rho_kappa)
        }
        
        # rho-h correlation
        r_rho_h, p_rho_h = stats.pearsonr(df['rho'], df['h'])
        stats_results['rho_h_correlation'] = {
            'r': float(r_rho_h),
            'p': float(p_rho_h)
        }
        
        # Prediction accuracy
        if 'pred_kappa' in df.columns:
            kappa_rmse = np.sqrt(np.mean((df['kappa'] - df['pred_kappa'])**2))
            stats_results['prediction_accuracy'] = {
                'kappa_rmse': float(kappa_rmse),
                'kappa_mape': float(np.mean(np.abs((df['kappa'] - df['pred_kappa']) / (df['pred_kappa'] + 1e-8))) * 100)
            }
    except Exception as e:
        print(f"[WARNING] Statistical computation error: {e}")
    
    return stats_results

def plot_correlation(df, output_dir='results'):
    """Correlation plots"""
    if plt is None:
        print("[WARNING] matplotlib not available, skipping plots.")
        return
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # kappa-h
        axes[0].scatter(df['kappa'], df['h'], s=100, alpha=0.6, c='blue')
        for i, row in df.iterrows():
            axes[0].annotate(f"ID{row['id']}", (row['kappa'], row['h']), 
                            fontsize=8, alpha=0.7)
        z = np.polyfit(df['kappa'], df['h'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['kappa'].min(), df['kappa'].max(), 100)
        axes[0].plot(x_line, p(x_line), "r--", alpha=0.8)
        axes[0].set_xlabel('Semantic Curvature (kappa)', fontsize=12)
        axes[0].set_ylabel('Hallucination Rate (h)', fontsize=12)
        axes[0].set_title('kappa vs h', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # rho-kappa
        axes[1].scatter(df['rho'], df['kappa'], s=100, alpha=0.6, c='green')
        for i, row in df.iterrows():
            axes[1].annotate(f"ID{row['id']}", (row['rho'], row['kappa']), 
                            fontsize=8, alpha=0.7)
        axes[1].set_xlabel('Information Density (rho)', fontsize=12)
        axes[1].set_ylabel('Semantic Curvature (kappa)', fontsize=12)
        axes[1].set_title('rho vs kappa', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # rho-h
        axes[2].scatter(df['rho'], df['h'], s=100, alpha=0.6, c='orange')
        for i, row in df.iterrows():
            axes[2].annotate(f"ID{row['id']}", (row['rho'], row['h']), 
                            fontsize=8, alpha=0.7)
        axes[2].set_xlabel('Information Density (rho)', fontsize=12)
        axes[2].set_ylabel('Hallucination Rate (h)', fontsize=12)
        axes[2].set_title('rho vs h', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{output_dir}/correlation_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Correlation plot saved: {output_path}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plot generation failed: {e}")

def plot_prediction_accuracy(df, output_dir='results'):
    """Prediction vs observation plots"""
    if plt is None:
        print("[WARNING] matplotlib not available, skipping plots.")
        return
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = [('rho', 'pred_rho', 'rho'), 
                   ('kappa', 'pred_kappa', 'kappa'), 
                   ('h', 'pred_h', 'h')]
        
        for idx, (obs, pred, label) in enumerate(metrics):
            if pred not in df.columns:
                continue
            
            axes[idx].scatter(df[pred], df[obs], s=100, alpha=0.6)
            
            # Diagonal (perfect prediction)
            min_val = min(df[pred].min(), df[obs].min())
            max_val = max(df[pred].max(), df[obs].max())
            axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect')
            
            for i, row in df.iterrows():
                axes[idx].annotate(f"ID{row['id']}", (row[pred], row[obs]), 
                                fontsize=8, alpha=0.7)
            
            axes[idx].set_xlabel(f'Predicted {label}', fontsize=12)
            axes[idx].set_ylabel(f'Observed {label}', fontsize=12)
            axes[idx].set_title(f'Prediction Accuracy: {label}', fontsize=14, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = f'{output_dir}/prediction_accuracy.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] Prediction accuracy plot saved: {output_path}")
        plt.close()
    except Exception as e:
        print(f"[WARNING] Plot generation failed: {e}")

def generate_latex_table(summary):
    """Generate LaTeX table"""
    latex = r"""\begin{table}[h]
\centering
\caption{Theoretical Predictions vs. Empirical Observations}
\begin{tabular}{clcccccc}
\hline
\textbf{ID} & \textbf{Category} & \boldmath$\rho_{\text{pred}}$ & \boldmath$\rho_{\text{obs}}$ & \boldmath$\kappa_{\text{pred}}$ & \boldmath$\kappa_{\text{obs}}$ & \boldmath$h_{\text{pred}}$ & \boldmath$h_{\text{obs}}$ \\
\hline
"""
    
    for _, row in summary.iterrows():
        category = row['category'][:15] if 'category' in row else 'unknown'
        pred_rho = row.get('pred_rho', 0)
        pred_kappa = row.get('pred_kappa', 0)
        pred_h = row.get('pred_h', 0)
        
        latex += f"{row['id']} & {category} & {pred_rho:.0f} & {row['rho']:.2f} & {pred_kappa:.2f} & {row['kappa']:.2f} & {pred_h:.2f} & {row['h']:.2f} \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\label{tab:results}
\end{table}
"""
    
    return latex

def main():
    print("="*60)
    print("Step 4: Analyzing Results")
    print("="*60)
    
    # Load data
    df = load_results()
    print(f"\n[SUCCESS] Loaded {len(df)} cases")
    
    # Summary table
    summary = create_summary_table(df)
    print("\n" + "="*60)
    print("Summary Table:")
    print("="*60)
    print(summary.to_string(index=False))
    
    # Statistical analysis
    stats_results = compute_statistics(df)
    print("\n" + "="*60)
    print("Statistical Analysis:")
    print("="*60)
    for key, value in stats_results.items():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    
    # Visualization
    print("\n" + "="*60)
    print("[*] Generating Visualizations...")
    print("="*60)
    plot_correlation(df)
    plot_prediction_accuracy(df)
    
    # LaTeX table
    latex_table = generate_latex_table(summary)
    try:
        with open('results/latex_table.txt', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print("[SUCCESS] LaTeX table saved: results/latex_table.txt")
    except Exception as e:
        print(f"[WARNING] LaTeX table save failed: {e}")
    
    # Save statistics
    try:
        with open('results/04_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats_results, f, indent=2)
        print("[SUCCESS] Statistics saved: results/04_statistics.json")
    except Exception as e:
        print(f"[WARNING] Statistics save failed: {e}")
    
    # Save CSV
    try:
        summary.to_csv('results/summary_table.csv', index=False)
        print("[SUCCESS] Summary table saved: results/summary_table.csv")
    except Exception as e:
        print(f"[WARNING] CSV save failed: {e}")
    
    print("\n" + "="*60)
    print("[SUCCESS] Step 4 Complete: All analyses finished")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/correlation_plots.png")
    print("  - results/prediction_accuracy.png")
    print("  - results/latex_table.txt")
    print("  - results/04_statistics.json")
    print("  - results/summary_table.csv")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

