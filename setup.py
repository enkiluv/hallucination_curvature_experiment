"""
Environment Setup Script (Windows 10) - Defensive Version
Run: python setup.py
"""
import subprocess
import sys
import os
import warnings
warnings.filterwarnings('ignore')

def run_command(command, description, critical=True):
    """Execute command with error handling"""
    print(f"\n{'='*60}")
    print(f"[*] {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print(f"[SUCCESS] {description}")
            if result.stdout:
                print(result.stdout[:500])  # First 500 chars only
            return True
        else:
            print(f"[WARNING] {description}")
            if result.stderr:
                print(result.stderr[:500])
            if critical:
                print("[ERROR] Failed at critical step.")
                return False
            return True
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} (exceeded 5 minutes)")
        return not critical
    except Exception as e:
        print(f"[ERROR] {description} - {e}")
        return not critical

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"[INFO] Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("[ERROR] Python 3.8 or higher is required.")
        return False
    
    if version < (3, 10):
        print("[WARNING] Python 3.10+ recommended (current version works)")
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ['data', 'src', 'results', 'logs']
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"[SUCCESS] Directory created: {d}/")
        except Exception as e:
            print(f"[ERROR] Failed to create directory: {d}/ - {e}")
            return False
    return True

def install_packages():
    """Install packages"""
    print("\n" + "="*60)
    print("[*] Installing packages...")
    print("="*60)
    
    # Upgrade pip
    if not run_command(
        f'"{sys.executable}" -m pip install --upgrade pip',
        "Upgrading pip",
        critical=False
    ):
        print("[WARNING] pip upgrade failed but continuing.")
    
    # Install from requirements.txt or manually
    if not os.path.exists('requirements.txt'):
        print("[WARNING] requirements.txt not found. Installing manually.")
        packages = [
            'torch==2.0.1',
            'transformers==4.35.0',
            'sentence-transformers==2.2.2',
            'faiss-cpu==1.7.4',
            'numpy==1.24.3',
            'scipy==1.11.3',
            'pandas==2.0.3',
            'matplotlib==3.7.2',
            'seaborn==0.12.2',
            'nltk==3.8.1',
            'tqdm==4.66.1'
        ]
        
        for pkg in packages:
            print(f"\n[*] Installing: {pkg}")
            run_command(
                f'"{sys.executable}" -m pip install {pkg}',
                f"Installing: {pkg}",
                critical=False
            )
    else:
        run_command(
            f'"{sys.executable}" -m pip install -r requirements.txt',
            "Installing from requirements.txt",
            critical=True
        )
    
    return True

def download_models():
    """Download NLP models"""
    print("\n" + "="*60)
    print("[*] Downloading NLP models...")
    print("="*60)
    
    # NLTK data
    try:
        import nltk
        print("[*] Downloading NLTK data...")
        
        downloads = ['wordnet', 'omw-1.4', 'punkt', 'averaged_perceptron_tagger']
        for item in downloads:
            try:
                nltk.download(item, quiet=True)
                print(f"  [SUCCESS] {item}")
            except Exception as e:
                print(f"  [WARNING] Failed to download {item}: {e}")
        
        print("[SUCCESS] NLTK data download complete")
    except ImportError:
        print("[WARNING] Cannot import NLTK. Install manually later.")
    except Exception as e:
        print(f"[WARNING] NLTK download error: {e}")
    
    # spaCy model (optional)
    print("\n[*] Attempting to download spaCy model...")
    spacy_result = run_command(
        f'"{sys.executable}" -m spacy download en_core_web_sm',
        "spaCy English model",
        critical=False
    )
    
    if not spacy_result:
        print("[WARNING] spaCy model download failed. Some features may be limited.")
    
    return True

def verify_installation():
    """Verify installation"""
    print("\n" + "="*60)
    print("[*] Verifying installation...")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence-Transformers',
        'faiss': 'FAISS',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib'
    }
    
    all_ok = True
    for pkg, name in required_packages.items():
        try:
            __import__(pkg)
            print(f"  [SUCCESS] {name}")
        except ImportError:
            print(f"  [ERROR] {name} - Installation failed")
            all_ok = False
    
    return all_ok

def main():
    print("""
    ================================================================
    
         Hallucination Curvature Experiment - Setup
         Windows 10 + Python 3.10 (Defensive Version)
    
    ================================================================
    """)
    
    # 1. Check Python version
    if not check_python_version():
        return
    
    # 2. Create virtual environment (optional)
    create_venv = input("\nCreate virtual environment? (y/n, recommended: y): ").lower()
    if create_venv == 'y':
        print("\n[*] Creating virtual environment...")
        if run_command('python -m venv venv', "Creating virtual environment", critical=False):
            print("""
            [SUCCESS] Virtual environment created!
            
            Activate with:
              Windows: venv\\Scripts\\activate
              
            After activation, run this script again:
              python setup.py
            """)
            return
        else:
            print("[WARNING] Virtual environment creation failed. Continuing.")
    
    # 3. Create directories
    if not create_directories():
        print("[ERROR] Directory creation failed. Create manually.")
        return
    
    # 4. Install packages
    if not install_packages():
        print("[ERROR] Package installation failed.")
        return
    
    # 5. Download models
    download_models()
    
    # 6. Verify installation
    if not verify_installation():
        print("""
        [WARNING] Some package verification failed
        
        Install manually:
          pip install torch transformers sentence-transformers faiss-cpu
        """)
        return
    
    print("""
    ================================================================
    
         [SUCCESS] Installation Complete!
    
         Next steps:
         1. Check data/test_inputs.json file
         2. Run: python run_all.py
    
    ================================================================
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Interrupted by user.")
    except Exception as e:
        print(f"\n\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

