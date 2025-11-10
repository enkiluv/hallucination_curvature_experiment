"""
Hallucination Rate (h) Computation - Defensive Version
"""
import json
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
        return None

def safe_import_transformers():
    """Safe Transformers import"""
    try:
        from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
        return pipeline, GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("[ERROR] Cannot import Transformers.")
        return None, None, None

def safe_import_sentence_transformers():
    """Safe Sentence-Transformers import"""
    try:
        from sentence_transformers import SentenceTransformer, util
        return SentenceTransformer, util
    except ImportError:
        print("[ERROR] Cannot import Sentence-Transformers.")
        return None, None

# Import dependencies
torch = safe_import_torch()
if torch is None:
    sys.exit(1)

pipeline, GPT2LMHeadModel, GPT2Tokenizer = safe_import_transformers()
if pipeline is None:
    sys.exit(1)

SentenceTransformer, util = safe_import_sentence_transformers()
if SentenceTransformer is None:
    sys.exit(1)

def generate_responses(text, model_name='gpt2', num_samples=5):
    """
    Generate multiple responses for the same input
    """
    print(f"    [*] Generating {num_samples} responses...")
    
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Use GPU if available
        device = 0 if torch.cuda.is_available() else -1
        generator = pipeline(
            'text-generation', 
            model=model, 
            tokenizer=tokenizer, 
            device=device
        )
        
        responses = []
        
        for i in range(num_samples):
            try:
                output = generator(
                    text,
                    max_length=len(tokenizer.encode(text)) + 50,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                
                # Remove original prompt
                response = output[len(text):].strip()
                responses.append(response)
            except Exception as e:
                print(f"      [WARNING] Generation {i+1} failed: {e}")
                responses.append("[GENERATION FAILED]")
        
        print(f"      [SUCCESS] Generated {len(responses)} responses")
        return responses
    
    except Exception as e:
        print(f"    [ERROR] Response generation failed: {e}")
        return ["[ERROR]"] * num_samples

def evaluate_factuality_exact(response, ground_truth):
    """
    Exact match evaluation (when ground truth exists)
    """
    if not ground_truth:
        return None
    
    response_lower = response.lower()
    
    if isinstance(ground_truth, list):
        for gt in ground_truth:
            if gt.lower() in response_lower:
                return 0  # Correct
        return 1  # Hallucinated
    else:
        if ground_truth.lower() in response_lower:
            return 0
        return 1

def evaluate_factuality_semantic(response, ground_truth, threshold=0.7):
    """
    Semantic similarity-based evaluation
    """
    if not ground_truth:
        return None
    
    try:
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        emb1 = sbert.encode(response, convert_to_tensor=True)
        emb2 = sbert.encode(ground_truth, convert_to_tensor=True)
        
        similarity = util.cos_sim(emb1, emb2).item()
        
        return 1 if similarity < threshold else 0
    except Exception as e:
        print(f"      [WARNING] Semantic evaluation failed: {e}")
        return None

def evaluate_consistency(responses):
    """
    Self-consistency evaluation (when no ground truth)
    """
    if len(responses) < 2:
        return 0.5  # Unknown
    
    # Count most common response
    from collections import Counter
    response_counts = Counter(responses)
    most_common_count = response_counts.most_common(1)[0][1]
    consistency = most_common_count / len(responses)
    
    # Low consistency -> high hallucination probability
    h = 1 - consistency
    
    return h

def compute_hallucination(text, ground_truth=None, num_samples=5):
    """
    Compute hallucination rate
    """
    responses = generate_responses(text, num_samples=num_samples)
    
    if ground_truth:
        # Has ground truth: direct evaluation
        hallucination_scores = []
        for response in responses:
            score = evaluate_factuality_exact(response, ground_truth)
            if score is None:
                score = evaluate_factuality_semantic(response, ground_truth)
            if score is not None:
                hallucination_scores.append(score)
        
        if len(hallucination_scores) > 0:
            h = sum(hallucination_scores) / len(hallucination_scores)
        else:
            h = 0.5  # Unknown
        
        return {
            'h': h,
            'method': 'ground_truth',
            'responses': responses,
            'scores': hallucination_scores,
            'success': True,
            'error': None
        }
    else:
        # No ground truth: consistency-based
        h = evaluate_consistency(responses)
        
        return {
            'h': h,
            'method': 'consistency',
            'responses': responses,
            'consistency': 1 - h,
            'success': True,
            'error': None
        }

def main():
    print("="*60)
    print("Step 3: Computing Hallucination Rate (h)")
    print("="*60)
    
    # Load previous results
    prev_results_path = 'results/02_kappa_results.json'
    if not os.path.exists(prev_results_path):
        print(f"[ERROR] File not found: {prev_results_path}")
        print("   Run 02_compute_kappa.py first.")
        sys.exit(1)
    
    try:
        with open(prev_results_path, 'r', encoding='utf-8') as f:
            kappa_results = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load previous results: {e}")
        sys.exit(1)
    
    # Load test data for ground truth
    test_data_path = 'data/test_inputs.json'
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load test data: {e}")
        test_data = {'test_cases': []}
    
    # Map ground truths
    ground_truths = {
        case['id']: case.get('ground_truth') 
        for case in test_data.get('test_cases', [])
    }
    
    results = []
    successful = 0
    failed = 0
    
    for case in kappa_results:
        case_id = case['id']
        text = case['text']
        ground_truth = ground_truths.get(case_id)
        
        print(f"\n[{case_id}/5] {text[:50]}...")
        
        try:
            h_result = compute_hallucination(text, ground_truth, num_samples=5)
            
            if h_result['success']:
                successful += 1
                print(f"  [SUCCESS] h = {h_result['h']:.4f} (method: {h_result['method']})")
            else:
                failed += 1
                print(f"  [ERROR] Computation failed")
            
            result = {
                **case,
                'h': h_result['h'],
                'h_method': h_result['method'],
                'responses': h_result['responses']
            }
            
            if 'scores' in h_result:
                result['h_scores'] = h_result['scores']
            if 'consistency' in h_result:
                result['consistency'] = h_result['consistency']
            
            results.append(result)
        
        except Exception as e:
            print(f"  [ERROR] Case processing error: {e}")
            failed += 1
            # Add placeholder result
            result = {
                **case,
                'h': 0.5,
                'h_method': 'error',
                'responses': [],
                'error': str(e)
            }
            results.append(result)
    
    # Save
    output_path = 'results/03_final_results.json'
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[SUCCESS] Results saved: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Save failed: {e}")
    
    print("\n" + "="*60)
    print(f"[SUCCESS] Step 3 Complete")
    print(f"   Success: {successful}/{len(results)}")
    if failed > 0:
        print(f"   Failed: {failed}/{len(results)}")
    print("="*60)

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

