"""
Semantic Curvature (kappa) Computation - Improved Version
"""
import json
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def safe_import_torch():
    """Safe PyTorch import"""
    try:
        import torch
        return torch
    except ImportError:
        print("[ERROR] Cannot import PyTorch.")
        print("   Install: pip install torch")
        return None

def safe_import_transformers():
    """Safe Transformers import"""
    try:
        from transformers import GPT2Model, GPT2Tokenizer
        return GPT2Model, GPT2Tokenizer
    except ImportError:
        print("[ERROR] Cannot import Transformers.")
        return None, None

def safe_import_sentence_transformers():
    """Safe Sentence-Transformers import"""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        print("[ERROR] Cannot import Sentence-Transformers.")
        return None

def safe_import_faiss():
    """Safe FAISS import"""
    try:
        import faiss
        return faiss
    except ImportError:
        print("[ERROR] Cannot import FAISS.")
        print("   Install: pip install faiss-cpu")
        return None

# Import all dependencies
torch = safe_import_torch()
if torch is None:
    sys.exit(1)

GPT2Model, GPT2Tokenizer = safe_import_transformers()
if GPT2Model is None:
    sys.exit(1)

SentenceTransformer = safe_import_sentence_transformers()
if SentenceTransformer is None:
    sys.exit(1)

faiss = safe_import_faiss()
if faiss is None:
    sys.exit(1)

from tqdm import tqdm

def generate_simple_neighbors(text, k=10):
    """
    Simple neighbor generation (fallback if paraphrasing fails)
    """
    neighbors = []
    
    # Method 1: Word order permutation
    words = text.split()
    if len(words) > 3:
        # Swap last two words
        variant = words.copy()
        variant[-1], variant[-2] = variant[-2], variant[-1]
        neighbors.append(' '.join(variant))
    
    # Method 2: Add phrases
    neighbors.append(f"{text} Please explain.")
    neighbors.append(f"Question: {text}")
    neighbors.append(f"{text} Thanks.")
    
    # Method 3: Case variations
    neighbors.append(text.lower())
    neighbors.append(text.upper())
    neighbors.append(text.title())
    
    # Method 4: Punctuation variations
    neighbors.append(text + "?")
    neighbors.append(text + ".")
    neighbors.append(text.replace("?", "."))
    
    # Remove duplicates
    neighbors = list(set([n for n in neighbors if n != text]))
    
    # Return k neighbors
    return neighbors[:k]

def generate_semantic_neighbors(text, k=10):
    """
    Semantic neighbor generation - Defensive version
    """
    print(f"    [*] Generating {k} semantic neighbors...")
    
    try:
        # Try T5 paraphraser
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        
        model_name = "ramsrigouthamg/t5_paraphraser"
        print(f"      [*] Loading T5 model...")
        
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        except Exception as e:
            print(f"      [WARNING] T5 model load failed: {e}")
            raise
        
        input_text = f"paraphrase: {text}"
        inputs = tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        )
        
        print(f"      [*] Generating paraphrases...")
        outputs = model.generate(
            inputs,
            max_length=128,
            num_return_sequences=min(k, 5),
            num_beams=min(k+2, 7),
            temperature=0.7,
            do_sample=True,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        neighbors = []
        for output in outputs:
            paraphrase = tokenizer.decode(output, skip_special_tokens=True)
            if paraphrase != text and paraphrase not in neighbors:
                neighbors.append(paraphrase)
        
        print(f"      [SUCCESS] Generated {len(neighbors)} paraphrases")
        
        # Add simple variants if insufficient
        if len(neighbors) < k:
            print(f"      [*] Adding simple variants...")
            simple = generate_simple_neighbors(text, k - len(neighbors))
            neighbors.extend(simple)
        
        return neighbors[:k]
    
    except Exception as e:
        print(f"      [WARNING] Paraphrasing failed: {e}")
        print(f"      [*] Using simple variants instead")
        return generate_simple_neighbors(text, k)

def compute_kappa_NPR(text, model_name='gpt2', k=10):
    """
    NPR-based curvature - Defensive version
    """
    print(f"    [*] Computing kappa_NPR (k={k})...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"      [INFO] Device: {device}")
        
        # Load model
        try:
            model = GPT2Model.from_pretrained(model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"      [WARNING] Model loading failed: {e}")
            raise
        
        # Generate neighbors
        try:
            neighbors = generate_semantic_neighbors(text, k=k)
            if len(neighbors) < k:
                print(f"      [WARNING] Only {len(neighbors)} neighbors generated (target: {k})")
        except Exception as e:
            print(f"      [WARNING] Neighbor generation failed: {e}")
            neighbors = generate_simple_neighbors(text, k)
        
        # SBERT embeddings
        try:
            print(f"      [*] Computing input embeddings...")
            sbert = SentenceTransformer('all-MiniLM-L6-v2')
            all_texts = [text] + neighbors
            X_embeddings = sbert.encode(
                all_texts, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
        except Exception as e:
            print(f"      [ERROR] SBERT embedding failed: {e}")
            raise
        
        # GPT-2 embeddings
        try:
            print(f"      [*] Computing latent embeddings...")
            Z_embeddings = []
            
            with torch.no_grad():
                for txt in all_texts:
                    try:
                        inputs = tokenizer(
                            txt, 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=128
                        ).to(device)
                        
                        outputs = model(**inputs)
                        z = outputs.last_hidden_state[0, -1, :].cpu().numpy()
                        Z_embeddings.append(z)
                    except Exception as e:
                        print(f"      [WARNING] Encoding failed for one text: {e}")
                        # Fallback: zero vector
                        if len(Z_embeddings) > 0:
                            Z_embeddings.append(np.zeros_like(Z_embeddings[0]))
                        else:
                            raise
            
            Z_embeddings = np.array(Z_embeddings)
        except Exception as e:
            print(f"      [ERROR] GPT-2 embedding failed: {e}")
            raise
        
        # Build FAISS indices
        try:
            print(f"      [*] Building k-NN indices...")
            d_X = X_embeddings.shape[1]
            d_Z = Z_embeddings.shape[1]
            
            index_X = faiss.IndexFlatL2(d_X)
            index_X.add(X_embeddings.astype('float32'))
            
            index_Z = faiss.IndexFlatL2(d_Z)
            index_Z.add(Z_embeddings.astype('float32'))
            
            # Find k-NN
            _, neighbors_X = index_X.search(
                X_embeddings[0:1].astype('float32'), k+1
            )
            neighbors_X_set = set(neighbors_X[0][1:])  # Exclude self
            
            _, neighbors_Z = index_Z.search(
                Z_embeddings[0:1].astype('float32'), k+1
            )
            neighbors_Z_set = set(neighbors_Z[0][1:])
            
            # Compute NPR
            intersection = len(neighbors_X_set & neighbors_Z_set)
            NPR = intersection / k
            kappa_NPR = 1 - NPR
            
            print(f"      [SUCCESS] kappa_NPR = {kappa_NPR:.4f} (NPR = {NPR:.4f})")
            
            return {
                'kappa_NPR': kappa_NPR,
                'NPR': NPR,
                'neighbors_count': len(neighbors),
                'success': True,
                'error': None
            }
        
        except Exception as e:
            print(f"      [ERROR] NPR computation failed: {e}")
            raise
    
    except Exception as e:
        print(f"    [ERROR] kappa_NPR computation error: {e}")
        return {
            'kappa_NPR': 0.0,
            'NPR': 1.0,
            'neighbors_count': 0,
            'success': False,
            'error': str(e)
        }

def compute_kappa_LLD(text, model_name='gpt2', epsilon=1e-3):
    """
    LLD-based curvature - IMPROVED VERSION with normalization
    """
    print(f"    [*] Computing kappa_LLD (Hessian approximation)...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GPT2Model.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(device)
        
        # Get center point embedding
        with torch.no_grad():
            outputs = model(**inputs)
            z_center = outputs.last_hidden_state[0, -1, :]
            embedding_dim = z_center.shape[0]
        
        # Finite difference Hessian approximation
        input_ids = inputs['input_ids'][0]
        n_tokens = len(input_ids)
        
        gradients = []
        # Sample more tokens for better estimate
        sample_size = min(5, n_tokens)  # Increased from 3 to 5
        
        for i in range(sample_size):
            # Slightly perturb token
            perturbed_ids = input_ids.clone()
            original_id = perturbed_ids[i].item()
            
            # Change to adjacent token
            perturbed_ids[i] = min(original_id + 1, tokenizer.vocab_size - 1)
            
            with torch.no_grad():
                perturbed_outputs = model(input_ids=perturbed_ids.unsqueeze(0))
                z_perturbed = perturbed_outputs.last_hidden_state[0, -1, :]
            
            # Compute difference
            diff = (z_perturbed - z_center).cpu().numpy()
            gradient_norm = np.linalg.norm(diff)
            gradients.append(gradient_norm)
        
        # LLD as variance of gradients
        if len(gradients) > 0:
            kappa_LLD_raw = float(np.std(gradients))
            
            # CRITICAL FIX: Normalize by embedding dimension
            # Typical gradient norms scale with sqrt(d)
            kappa_LLD = kappa_LLD_raw / np.sqrt(embedding_dim)
            
            print(f"      [SUCCESS] kappa_LLD = {kappa_LLD:.4f} (raw={kappa_LLD_raw:.2f}, normalized)")
        else:
            kappa_LLD = 0.0
            kappa_LLD_raw = 0.0
        
        return {
            'kappa_LLD': kappa_LLD,
            'kappa_LLD_raw': kappa_LLD_raw,
            'gradient_samples': len(gradients),
            'embedding_dim': embedding_dim,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        print(f"    [ERROR] kappa_LLD computation error: {e}")
        return {
            'kappa_LLD': 0.0,
            'kappa_LLD_raw': 0.0,
            'gradient_samples': 0,
            'embedding_dim': 0,
            'success': False,
            'error': str(e)
        }

def compute_kappa(text, model_name='gpt2', w1=0.6, w2=0.4):
    """
    Integrated curvature computation - IMPROVED VERSION
    """
    npr_result = compute_kappa_NPR(text, model_name)
    lld_result = compute_kappa_LLD(text, model_name)
    
    if npr_result['success'] and lld_result['success']:
        kappa = w1 * npr_result['kappa_NPR'] + w2 * lld_result['kappa_LLD']
        success = True
        error = None
    elif npr_result['success']:
        # Use NPR only
        kappa = npr_result['kappa_NPR']
        success = True
        error = "LLD failed, using NPR only"
    elif lld_result['success']:
        # Use LLD only
        kappa = lld_result['kappa_LLD']
        success = True
        error = "NPR failed, using LLD only"
    else:
        # Both failed
        kappa = 0.0
        success = False
        error = "Both NPR and LLD failed"
    
    return {
        'kappa': kappa,
        'kappa_NPR': npr_result['kappa_NPR'],
        'kappa_LLD': lld_result['kappa_LLD'],
        'kappa_LLD_raw': lld_result.get('kappa_LLD_raw', 0.0),
        'NPR': npr_result['NPR'],
        'embedding_dim': lld_result.get('embedding_dim', 0),
        'w1': w1,
        'w2': w2,
        'success': success,
        'error': error
    }

def main():
    print("="*60)
    print("Step 2: Computing Semantic Curvature (kappa) - IMPROVED")
    print("="*60)
    
    # Load previous results
    prev_results_path = 'results/01_rho_results.json'
    if not os.path.exists(prev_results_path):
        print(f"[ERROR] File not found: {prev_results_path}")
        print("   Run 01_compute_rho.py first.")
        sys.exit(1)
    
    try:
        with open(prev_results_path, 'r', encoding='utf-8') as f:
            rho_results = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load previous results: {e}")
        sys.exit(1)
    
    results = []
    successful = 0
    failed = 0
    
    for case in tqdm(rho_results, desc="Computing kappa"):
        case_id = case['id']
        text = case['text']
        
        print(f"\n[{case_id}/5] {text[:50]}...")
        
        kappa_result = compute_kappa(text)
        
        if kappa_result['success']:
            successful += 1
            print(f"  [SUCCESS] kappa = {kappa_result['kappa']:.4f}")
            print(f"     kappa_NPR = {kappa_result['kappa_NPR']:.4f}")
            print(f"     kappa_LLD = {kappa_result['kappa_LLD']:.4f}")
        else:
            failed += 1
            print(f"  [ERROR] Computation failed: {kappa_result['error']}")
        
        result = {
            **case,
            **kappa_result
        }
        
        results.append(result)
    
    # Save
    output_path = 'results/02_kappa_results.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Results saved: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Save failed: {e}")
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Step 2 Complete")
    print(f"   Success: {successful}/{len(results)}")
    if failed > 0:
        print(f"   Failed: {failed}/{len(results)}")
    print("="*60)
    
    # Summary statistics
    if results:
        kappas = [r['kappa'] for r in results if r['success']]
        if kappas:
            print(f"\n[SUMMARY] kappa statistics:")
            print(f"   Min: {min(kappas):.4f}")
            print(f"   Max: {max(kappas):.4f}")
            print(f"   Mean: {sum(kappas)/len(kappas):.4f}")

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

