"""
Complete Experiment Pipeline Execution
"""
import subprocess
import sys
import time
import os

def run_script(script_path, description):
    """Execute script"""
    print("\n" + "="*80)
    print(f"[*] {description}")
    print("="*80)
    
    start_time = time.time()
    
    result = subprocess.run([sys.executable, script_path], 
                          capture_output=False, text=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] {description} - Complete (Time: {elapsed:.1f}s)")
        return True
    else:
        print(f"\n[ERROR] {description} - Failed")
        return False

def main():
    print("""
    ================================================================
    
         Hallucination is Curvature: Pilot Experiment
    
         Complete Experiment Pipeline Execution
    
    ================================================================
    """)
    
    # Environment check
    print("[INFO] Environment Check:")
    print(f"   Python: {sys.version}")
    print(f"   Working Directory: {os.getcwd()}")
    
    # Check required files
    if not os.path.exists('data/test_inputs.json'):
        print("\n[ERROR] data/test_inputs.json not found!")
        return
    
    print("\n[SUCCESS] All required files verified")
    
    input("\nPress Enter to continue...")
    
    start_time = time.time()
    
    # Step 1: Compute rho
    if not run_script('src/01_compute_rho.py', "Step 1: Computing rho"):
        return
    
    # Step 2: Compute kappa
    if not run_script('src/02_compute_kappa.py', "Step 2: Computing kappa"):
        return
    
    # Step 3: Compute h
    if not run_script('src/03_compute_h.py', "Step 3: Computing h"):
        return
    
    # Step 4: Analyze
    if not run_script('src/04_analyze_results.py', "Step 4: Analyzing Results"):
        return
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("[SUCCESS] Experiment Complete!")
    print("="*80)
    print(f"Total Time: {total_time/60:.1f} minutes")
    print("\nGenerated Files:")
    print("  - results/correlation_plots.png")
    print("  - results/prediction_accuracy.png")
    print("  - results/latex_table.txt")
    print("  - results/summary_table.csv")
    print("  - results/04_statistics.json")
    print("\nNext Step: Add results to paper Section 4")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

