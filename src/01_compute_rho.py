"""
Information Density (rho) Computation - Corrected Version
"""
import json
import gzip
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def safe_import(package_name, pip_name=None):
    """Safe import with fallback"""
    try:
        return __import__(package_name)
    except ImportError:
        print(f"[WARNING] Cannot import {package_name}.")
        if pip_name:
            print(f"   Install: pip install {pip_name}")
        return None

# Import with error handling
transformers = safe_import('transformers', 'transformers')
if transformers is None:
    print("[ERROR] transformers is required. Install and retry.")
    sys.exit(1)

from transformers import GPT2Tokenizer

def compute_entropy_gzip(text):
    """Estimate entropy using gzip compression"""
    try:
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes, compresslevel=9)
        
        if len(text_bytes) == 0:
            return {
                'original_bytes': 0,
                'compressed_bytes': 0,
                'compression_ratio': 1.0,
                'entropy_bits_per_byte': 8.0
            }
        
        compression_ratio = len(compressed) / len(text_bytes)
        entropy_bits = 8 * compression_ratio
        
        return {
            'original_bytes': len(text_bytes),
            'compressed_bytes': len(compressed),
            'compression_ratio': compression_ratio,
            'entropy_bits_per_byte': entropy_bits
        }
    except Exception as e:
        print(f"  [WARNING] Compression error: {e}")
        # Fallback: assume maximum entropy
        return {
            'original_bytes': len(text.encode('utf-8')),
            'compressed_bytes': len(text.encode('utf-8')),
            'compression_ratio': 1.0,
            'entropy_bits_per_byte': 8.0
        }

def compute_rho(text, model_name='gpt2'):
    """
    Compute information density rho - CORRECTED VERSION
    
    Key fix: Use training corpus size, not just prompt size
    """
    try:
        # Load tokenizer
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"  [WARNING] Tokenizer load failed: {e}")
            print(f"  [*] Attempting online download...")
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, force_download=True)
        
        # Model capacity
        num_parameters = 124_000_000
        bits_per_parameter = 3.6  # Morris et al. (2025)
        C_model = num_parameters * bits_per_parameter
        
        # Estimate entropy from prompt (as proxy for domain)
        entropy_info = compute_entropy_gzip(text)
        
        # Tokenization
        try:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            num_tokens = len(tokens)
        except Exception as e:
            print(f"  [WARNING] Tokenization error: {e}")
            # Fallback: rough estimate
            num_tokens = max(1, len(text.split()))
        
        if num_tokens == 0:
            num_tokens = 1  # Prevent division by zero
        
        # bits per token for this specific input
        H_per_token = entropy_info['entropy_bits_per_byte'] * (
            entropy_info['original_bytes'] / num_tokens
        )
        
        # CRITICAL FIX: Use training corpus size estimate
        # GPT-2 was trained on WebText (~40GB, ~10 billion tokens)
        estimated_training_tokens = 10_000_000_000
        
        # Normalize H_per_token relative to typical English (1.0-1.5 bits/token)
        # This accounts for domain-specific entropy differences
        typical_english_entropy = 1.2  # bits/token
        domain_factor = H_per_token / typical_english_entropy
        
        # Effective entropy considering domain complexity
        effective_H_per_token = min(H_per_token, typical_english_entropy * 2.0)  # Cap at 2x typical
        
        # Total data entropy (training corpus)
        H_data = effective_H_per_token * estimated_training_tokens
        
        # Compression ratio
        rho = H_data / C_model
        
        return {
            'rho': rho,
            'H_data': H_data,
            'C_model': C_model,
            'H_per_token': H_per_token,
            'effective_H_per_token': effective_H_per_token,
            'num_tokens': num_tokens,
            'estimated_training_tokens': estimated_training_tokens,
            'domain_factor': domain_factor,
            'entropy_info': entropy_info,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"  [ERROR] rho computation error: {e}")
        return {
            'rho': 0.0,
            'H_data': 0.0,
            'C_model': 446400000.0,
            'H_per_token': 0.0,
            'effective_H_per_token': 0.0,
            'num_tokens': 0,
            'estimated_training_tokens': 0,
            'domain_factor': 0.0,
            'entropy_info': {},
            'success': False,
            'error': str(e)
        }

def main():
    print("="*60)
    print("Step 1: Computing Information Density (rho) - CORRECTED")
    print("="*60)
    
    # Load data
    data_path = 'data/test_inputs.json'
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}")
        print("   Create data/test_inputs.json file.")
        sys.exit(1)
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] File read error: {e}")
        sys.exit(1)
    
    if 'test_cases' not in data:
        print("[ERROR] JSON structure error: missing 'test_cases' key.")
        sys.exit(1)
    
    results = []
    successful = 0
    failed = 0
    
    print("\n[INFO] Using training corpus estimate: 10 billion tokens")
    print("[INFO] Model capacity: 446.4 million bits")
    print("")
    
    for case in data['test_cases']:
        try:
            case_id = case.get('id', '?')
            text = case.get('text', '')
            
            if not text:
                print(f"\n[WARNING] [{case_id}] Empty text. Skipping.")
                continue
            
            print(f"\n[{case_id}/5] Processing: {text[:50]}...")
            
            rho_result = compute_rho(text)
            
            if rho_result['success']:
                successful += 1
                print(f"  [SUCCESS] rho = {rho_result['rho']:.2f}")
                print(f"     H_per_token = {rho_result['H_per_token']:.2f} bits/token")
                print(f"     Effective H = {rho_result['effective_H_per_token']:.2f} bits/token")
                print(f"     H_data (total) = {rho_result['H_data']/1e9:.2f} billion bits")
                print(f"     Domain factor = {rho_result['domain_factor']:.2f}x")
            else:
                failed += 1
                print(f"  [ERROR] Computation failed: {rho_result['error']}")
            
            result = {
                'id': case_id,
                'text': text,
                'category': case.get('category', 'unknown'),
                'predicted_rho': case.get('predicted_rho', 0),
                **rho_result
            }
            
            results.append(result)
        
        except Exception as e:
            print(f"  [ERROR] Case processing error: {e}")
            failed += 1
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/01_rho_results.json'
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Results saved: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Save failed: {e}")
        # Fallback: save to current directory
        try:
            with open('rho_results_backup.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [*] Backup saved: rho_results_backup.json")
        except:
            pass
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Step 1 Complete")
    print(f"   Success: {successful}/{len(results)}")
    if failed > 0:
        print(f"   Failed: {failed}/{len(results)}")
    print("="*60)
    
    # Summary statistics
    if results:
        rhos = [r['rho'] for r in results if r['success']]
        if rhos:
            print(f"\n[SUMMARY] rho statistics:")
            print(f"   Min: {min(rhos):.2f}")
            print(f"   Max: {max(rhos):.2f}")
            print(f"   Mean: {sum(rhos)/len(rhos):.2f}")

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

