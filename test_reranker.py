#!/usr/bin/env python3
"""
Unified Reranker Test Framework
==============================

Tests both BGE and Qwen reranker models using official Transformers implementations
and Ollama API. Supports multiple model types and configurations.

Usage:
    # Test all models
    uv run python test_reranker.py
    
    # Test specific model type
    uv run python test_reranker.py --model-type bge
    uv run python test_reranker.py --model-type qwen
    
    # Test specific implementation
    uv run python test_reranker.py --implementation official
    uv run python test_reranker.py --implementation ollama
    
    # Test specific model
    uv run python test_reranker.py --model BAAI/bge-reranker-v2-m3
    uv run python test_reranker.py --model qwen_reranker_v2

Environment Variables:
    MODEL_NAME: Override default model name for Ollama tests
"""

import json
import time
import os
import glob
import argparse
import requests
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from FlagEmbedding import FlagReranker
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    'bge': {
        'official': {
            'models': [
                "BAAI/bge-reranker-v2-m3",
                "BAAI/bge-reranker-base", 
                "BAAI/bge-reranker-large"
            ],
            'default': "BAAI/bge-reranker-v2-m3"
        },
        'ollama': {
            'models': [
                "bge-base",
                "bge-large", 
                "bge-v2-m3"
            ],
            'default': "bge-v2-m3"
        }
    },
    'qwen': {
        'official': {
            'models': [
                "Qwen/Qwen3-Reranker-0.6B",
                "Qwen/Qwen3-Reranker-4B", 
                "Qwen/Qwen3-Reranker-8B"
            ],
            'default': "Qwen/Qwen3-Reranker-0.6B"
        },
        'ollama': {
            'models': [
                "qwen3-0.6b",
                "qwen3-4b",
                "qwen3-8b"
            ],
            'default': "qwen3-0.6b"
        }
    }
}

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

def load_bge_model(model_name):
    """Load BGE reranker model using FlagEmbedding"""
    try:
        print(f"📦 Loading BGE reranker model: {model_name}")
        reranker = FlagReranker(model_name, use_fp16=True)
        return {
            'type': 'bge',
            'reranker': reranker,
            'model_name': model_name
        }, None
    except Exception as e:
        return None, str(e)

def load_qwen_model(model_name):
    """Load Qwen3 reranker model using Transformers"""
    try:
        print(f"📦 Loading Qwen3 reranker model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        
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
            'type': 'qwen',
            'tokenizer': tokenizer,
            'model': model,
            'token_false_id': token_false_id,
            'token_true_id': token_true_id,
            'max_length': max_length,
            'prefix_tokens': prefix_tokens,
            'suffix_tokens': suffix_tokens,
            'model_name': model_name
        }, None
    except Exception as e:
        return None, str(e)

def format_qwen_instruction(instruction, query, doc):
    """Format instruction for Qwen model"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )

def process_qwen_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length, model):
    """Process inputs for Qwen model"""
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

def compute_qwen_logits(inputs, model, token_true_id, token_false_id, **kwargs):
    """Compute logits for Qwen model"""
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def test_official_reranker(test_case, model_info):
    """Test official reranker implementation"""
    try:
        query = test_case["query"]
        documents = test_case["documents"]
        
        # Handle empty documents case
        if not documents:
            return {
                "success": True,
                "results": [],
                "time": 0,
                "error": None
            }
        
        start_time = time.time()
        
        if model_info['type'] == 'bge':
            # BGE reranker
            pairs = [[query, doc] for doc in documents]
            scores = model_info['reranker'].compute_score(pairs, normalize=True)
            
        elif model_info['type'] == 'qwen':
            # Qwen reranker
            instruction = test_case.get("instruction", "Given a web search query, retrieve relevant passages that answer the query")
            pairs = [format_qwen_instruction(instruction, query, doc) for doc in documents]
            
            inputs = process_qwen_inputs(
                pairs, 
                model_info['tokenizer'], 
                model_info['prefix_tokens'], 
                model_info['suffix_tokens'], 
                model_info['max_length'], 
                model_info['model']
            )
            
            scores = compute_qwen_logits(
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
                "relevance_score": float(score),
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

def test_ollama_reranker(test_case, model_name):
    """Test Ollama reranking API"""
    url = "http://localhost:11434/api/rerank"
    
    payload = {
        "model": model_name,
        "query": test_case["query"],
        "documents": test_case["documents"]
    }
    
    # Add optional parameters
    if "instruction" in test_case:
        payload["instruction"] = test_case["instruction"]
    if "top_n" in test_case:
        payload["top_n"] = test_case["top_n"]
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "results": result.get("results", []),
            "time": elapsed,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "time": time.time() - start_time,
            "error": str(e)
        }

def run_tests(model_type=None, implementation=None, specific_model=None):
    """Run tests based on configuration"""
    test_cases = load_test_cases()
    results = {}
    
    # Determine what to test
    if specific_model:
        # Test specific model
        if 'bge' in specific_model.lower():
            model_type = 'bge'
        elif 'qwen' in specific_model.lower():
            model_type = 'qwen'
        else:
            print(f"❌ Unknown model: {specific_model}")
            return
    
    if model_type and model_type not in MODEL_CONFIGS:
        print(f"❌ Unknown model type: {model_type}")
        return
    
    # Test configurations
    configs = []
    
    if model_type:
        # Test specific model type
        if implementation == 'official':
            for model in MODEL_CONFIGS[model_type]['official']['models']:
                configs.append((model_type, 'official', specific_model or model))
        elif implementation == 'ollama':
            for model in MODEL_CONFIGS[model_type]['ollama']['models']:
                configs.append((model_type, 'ollama', specific_model or model))
        else:
            # Test both implementations
            for model in MODEL_CONFIGS[model_type]['official']['models']:
                configs.append((model_type, 'official', specific_model or model))
            for model in MODEL_CONFIGS[model_type]['ollama']['models']:
                configs.append((model_type, 'ollama', specific_model or model))
    else:
        # Test all configurations
        for mt in MODEL_CONFIGS:
            if implementation == 'official':
                for model in MODEL_CONFIGS[mt]['official']['models']:
                    configs.append((mt, 'official', model))
            elif implementation == 'ollama':
                for model in MODEL_CONFIGS[mt]['ollama']['models']:
                    configs.append((mt, 'ollama', model))
            else:
                # Test both implementations
                for model in MODEL_CONFIGS[mt]['official']['models']:
                    configs.append((mt, 'official', model))
                for model in MODEL_CONFIGS[mt]['ollama']['models']:
                    configs.append((mt, 'ollama', model))
    
    # Run tests
    for model_type, impl, model_name in configs:
        print(f"\n🔧 Testing {model_type.upper()} {impl.upper()}: {model_name}")
        print("=" * 60)
        
        if impl == 'official':
            # Load model
            if model_type == 'bge':
                model_info, error = load_bge_model(model_name)
            else:
                model_info, error = load_qwen_model(model_name)
            
            if error:
                print(f"❌ Failed to load model: {error}")
                continue
            
            print(f"✅ Model loaded successfully")
            
            # Test all cases
            model_results = {}
            for test_case in test_cases:
                print(f"\n📋 Testing: {test_case['name']}")
                print(f"Query: {test_case['query']}")
                print(f"Documents: {len(test_case['documents'])}")
                
                result = test_official_reranker(test_case, model_info)
                model_results[test_case["name"]] = {
                    "test_case": test_case,
                    "result": result
                }
                
                # Print summary
                print(f"✅ {'SUCCESS' if result['success'] else 'FAILED'} ({result['time']:.3f}s)")
                
                if result.get("error"):
                    print(f"❌ Error: {result['error']}")
                
                if result["success"] and result["results"]:
                    print("📈 Rankings:")
                    for i, res in enumerate(result["results"]):
                        doc = res["document"]
                        score = res["relevance_score"]
                        print(f"  {i+1}. {doc[:50]}... (score: {score:.4f})")
            
            results[f"{model_type}_{impl}_{model_name.replace('/', '_')}"] = model_results
            
        else:  # ollama
            # Test all cases
            model_results = {}
            for test_case in test_cases:
                print(f"\n📋 Testing: {test_case['name']}")
                print(f"Query: {test_case['query']}")
                print(f"Documents: {len(test_case['documents'])}")
                
                result = test_ollama_reranker(test_case, model_name)
                
                # Check if this test is expected to fail
                expected_to_fail = test_case.get("_test_metadata", {}).get("expected_to_fail", False)
                
                # Determine if test passed based on expectations
                test_passed = False
                if expected_to_fail:
                    test_passed = not result['success']
                    status = "SUCCESS (Expected Failure)" if test_passed else "FAILED (Should Have Failed)"
                else:
                    test_passed = result['success']
                    status = "SUCCESS" if test_passed else "FAILED"
                
                model_results[test_case["name"]] = {
                    "test_case": test_case,
                    "result": result,
                    "test_passed": test_passed
                }
                
                # Print summary
                print(f"✅ {status} ({result['time']:.3f}s)")
                
                if result.get("error"):
                    if expected_to_fail:
                        print(f"✅ Expected Error: {result['error']}")
                    else:
                        print(f"❌ Error: {result['error']}")
                
                if result["success"] and result["results"]:
                    print("📈 Rankings:")
                    for i, res in enumerate(result["results"]):
                        doc = res["document"]
                        score = res["relevance_score"]
                        print(f"  {i+1}. {doc[:50]}... (score: {score:.4f})")
            
            results[f"{model_type}_{impl}_{model_name}"] = model_results
    
    return results

def save_results(results, model_type=None, implementation=None):
    """Save results to appropriate files"""
    os.makedirs("results", exist_ok=True)
    
    if model_type and implementation:
        # Save to specific file
        filename = f"results/{model_type}_{implementation}_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
    else:
        # Save to separate files by model type and implementation
        for key, result in results.items():
            filename = f"results/{key}_results.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Results saved to: {filename}")

def print_summary(results):
    """Print test summary"""
    print(f"\n📊 TEST SUMMARY")
    print("=" * 50)
    
    total_tests = 0
    successful_tests = 0
    
    for key, result in results.items():
        model_total = len(result)
        model_successful = sum(1 for r in result.values() if r["result"]["success"])
        total_tests += model_total
        successful_tests += model_successful
        
        print(f"\n📊 {key}:")
        print(f"  Total Tests: {model_total}")
        print(f"  Successful Tests: {model_successful}")
        print(f"  Success Rate: {model_successful/model_total*100:.1f}%")
    
    if total_tests > 0:
        print(f"\n📊 OVERALL SUMMARY")
        print("=" * 40)
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tests: {successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Unified Reranker Test Framework")
    parser.add_argument("--model-type", choices=["bge", "qwen"], help="Test specific model type")
    parser.add_argument("--implementation", choices=["official", "ollama"], help="Test specific implementation")
    parser.add_argument("--model", help="Test specific model name")
    
    args = parser.parse_args()
    
    print("🤖 UNIFIED RERANKER TEST FRAMEWORK")
    print("=" * 50)
    
    # Run tests
    results = run_tests(args.model_type, args.implementation, args.model)
    
    if results:
        # Save results
        save_results(results, args.model_type, args.implementation)
        
        # Print summary
        print_summary(results)
        
        print("\n✅ Tests completed successfully")
    else:
        print("❌ No tests were run")

if __name__ == "__main__":
    main() 