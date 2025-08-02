#!/usr/bin/env python3
"""
Official BGE-Reranker Test using Real Transformers Implementation
Tests the actual official BGE-Reranker models using Transformers library
"""

import json
import time
import os
import glob
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from FlagEmbedding import FlagReranker, FlagLLMReranker
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
                
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load {test_file}: {e}")
    
    return test_cases

def load_real_model(model_name="BAAI/bge-reranker-v2-m3"):
    """Load real BGE-Reranker model using Transformers"""
    try:
        print(f"📦 Loading real BGE-Reranker model: {model_name}")
        
        # All remaining models are normal BGE rerankers
        print("🔧 Using normal BGE reranker (FlagReranker)")
        reranker = FlagReranker(model_name, use_fp16=True)
        return {
            'type': 'normal',
            'reranker': reranker,
            'model_name': model_name
        }, None
        
    except Exception as e:
        return None, str(e)

def test_official_bge(test_case, model_info):
    """Test real BGE-Reranker using FlagEmbedding"""
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
        
        # Process documents
        start_time = time.time()
        
        # Create pairs for all documents
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores for normal BGE reranker
        scores = model_info['reranker'].compute_score(pairs, normalize=True)
        
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

def main():
    """Run real BGE-Reranker tests"""
    print("🤖 REAL BGE-RERANKER TEST (FlagEmbedding)")
    print("=" * 50)
    
    # Test different BGE models
    bge_models = [
        "BAAI/bge-reranker-v2-m3",
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-large"
    ]
    
    results = {}
    test_cases = load_test_cases()
    
    for model_name in bge_models:
        print(f"\n🔧 Testing model: {model_name}")
        
        # Load model
        model_info, error = load_real_model(model_name)
        if error:
            print(f"❌ Failed to load model {model_name}: {error}")
            continue
        
        print(f"✅ Model loaded successfully ({model_info['type']} type)")
        
        model_results = {}
        
        for test_case in test_cases:
            print(f"\n📋 Testing: {test_case['name']}")
            print(f"Query: {test_case['query']}")
            print(f"Documents: {len(test_case['documents'])}")
            
            # Test real implementation
            print(f"🤖 Testing Real BGE-Reranker ({model_name})...")
            real_result = test_official_bge(test_case, model_info)
            
            model_results[test_case["name"]] = {
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
        
        results[model_name] = model_results
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Save results
    output_file = "results/bge_official_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    total_tests = 0
    successful_tests = 0
    
    for model_name, model_results in results.items():
        model_total = len(model_results)
        model_successful = sum(1 for r in model_results.values() if r["result"]["success"])
        total_tests += model_total
        successful_tests += model_successful
        
        print(f"\n📊 {model_name}:")
        print(f"  Total Tests: {model_total}")
        print(f"  Successful Tests: {model_successful}")
        print(f"  Success Rate: {model_successful/model_total*100:.1f}%")
    
    print(f"\n📊 OVERALL SUMMARY")
    print("=" * 40)
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    print("✅ BGE tests completed")

if __name__ == "__main__":
    main() 