#!/usr/bin/env python3
"""
Official Qwen3-Reranker Test using Real Transformers Implementation
Tests the actual official Qwen3-Reranker model using Transformers library
"""

import json
import time
import os
import glob
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np

def load_test_cases():
    """Load test cases from JSON files in tests/ directory"""
    test_cases = []
    test_files = glob.glob("tests/test_*.json")
    
    for test_file in sorted(test_files):
        try:
            with open(test_file, 'r') as f:
                test_data = json.load(f)
                
            # Extract test case name from filename
            test_name = os.path.splitext(os.path.basename(test_file))[0]
            
            # Create test case structure
            test_case = {
                "name": test_name,
                "file": test_file,
                "query": test_data.get("query", ""),
                "documents": test_data.get("documents", [])
            }
            
            # Add optional parameters if present
            if "instruction" in test_data:
                test_case["instruction"] = test_data["instruction"]
            if "top_n" in test_data:
                test_case["top_n"] = test_data["top_n"]
            if "model" in test_data:
                test_case["model"] = test_data["model"]
            if "_test_metadata" in test_data:
                test_case["_test_metadata"] = test_data["_test_metadata"]
                
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load {test_file}: {e}")
    
    return test_cases

def load_real_model():
    """Load real Qwen3-Reranker model using Transformers"""
    try:
        print("📦 Loading real Qwen3-Reranker model...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
        
        # Get token IDs for yes/no
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        
        # Setup template tokens
        max_length = 8192
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
        return {
            'tokenizer': tokenizer,
            'model': model,
            'token_false_id': token_false_id,
            'token_true_id': token_true_id,
            'max_length': max_length,
            'prefix_tokens': prefix_tokens,
            'suffix_tokens': suffix_tokens
        }, None
        
    except Exception as e:
        return None, str(e)

def format_instruction(instruction, query, doc):
    """Format instruction for the model"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output

def process_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length, model):
    """Process inputs for the model"""
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

def compute_logits(inputs, model, token_true_id, token_false_id, **kwargs):
    """Compute logits and convert to probabilities"""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def test_official_qwen(test_case, model_info):
    """Test real Qwen3-Reranker using Transformers"""
    try:
        query = test_case["query"]
        documents = test_case["documents"]
        instruction = test_case.get("instruction", "Given a web search query, retrieve relevant passages that answer the query")
        
        # Handle empty documents case
        if not documents:
            return {
                "success": True,
                "results": [],
                "time": 0,
                "error": None
            }
        
        # Process documents
        start_time = time.time()
        
        # Create pairs for all documents
        pairs = [format_instruction(instruction, query, doc) for doc in documents]
        
        # Process inputs
        inputs = process_inputs(
            pairs, 
            model_info['tokenizer'], 
            model_info['prefix_tokens'], 
            model_info['suffix_tokens'], 
            model_info['max_length'], 
            model_info['model']
        )
        
        # Compute scores
        scores = compute_logits(
            inputs, 
            model_info['model'], 
            model_info['token_true_id'], 
            model_info['token_false_id']
        )
        
        elapsed = time.time() - start_time
        
        # Create results
        results = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                "index": idx,
                "document": doc,
                "relevance_score": score,
                "raw_response": f"{score:.4f}"
            })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Apply top_n if specified
        if "top_n" in test_case:
            results = results[:test_case["top_n"]]
        
        return {
            "success": True,
            "results": results,
            "time": elapsed,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "time": 0,
            "error": str(e)
        }

def main():
    """Run real Qwen3-Reranker tests"""
    print("🤖 REAL QWEN3-RERANKER TEST (Transformers)")
    print("=" * 50)
    
    # Load model once
    model_info, error = load_real_model()
    if error:
        print(f"❌ Failed to load model: {error}")
        return
    
    print("✅ Model loaded successfully")
    print(f"🎯 Token IDs: false={model_info['token_false_id']}, true={model_info['token_true_id']}")
    
    results = {}
    test_cases = load_test_cases()
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print(f"Documents: {len(test_case['documents'])}")
        
        # Test real implementation
        print("🤖 Testing Real Qwen3-Reranker...")
        real_result = test_official_qwen(test_case, model_info)
        
        results[test_case["name"]] = {
            "test_case": test_case,
            "result": real_result
        }
        
        # Print summary
        print(f"✅ Real: {'SUCCESS' if real_result['success'] else 'FAILED'} ({real_result['time']:.3f}s)")
        
        if real_result.get("error"):
            print(f"❌ Real Error: {real_result['error']}")
        
        if real_result["success"] and real_result["results"]:
            print("📈 Rankings:")
            for i, result in enumerate(real_result["results"]):
                doc = result["document"]
                score = result["relevance_score"]
                raw_response = result.get("raw_response", "")
                print(f"  {i+1}. {doc[:50]}... (score: {score:.4f})")
                if raw_response:
                    print(f"     Raw: {raw_response}")
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save results
    output_file = "results/qwen_official_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    successful_tests = sum(1 for r in results.values() if r["result"]["success"])
    total_tests = len(results)
    
    print(f"\n📊 SUMMARY")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    print("✅ Real tests completed")

if __name__ == "__main__":
    main() 