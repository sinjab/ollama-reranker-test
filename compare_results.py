#!/usr/bin/env python3
"""
Comprehensive Reranker Results Comparison
========================================

Compares results across all tested models (BGE and Qwen, Official and Ollama)
to analyze performance differences, ranking accuracy, and score distributions.
"""

import json
import os
import glob
from typing import Dict, List, Any
import numpy as np

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def compare_rankings(rankings1: List[Dict], rankings2: List[Dict], model1_name: str, model2_name: str) -> Dict[str, Any]:
    """Compare rankings between two models"""
    if not rankings1 or not rankings2:
        return {
            "ranking_match": False,
            "score_differences": [],
            "top_rank_match": False,
            "correlation": 0.0,
            "avg_score_difference": 0.0
        }
    
    # Create index to score mappings
    scores1 = {r["index"]: r["relevance_score"] for r in rankings1}
    scores2 = {r["index"]: r["relevance_score"] for r in rankings2}
    
    # Check if top rank matches
    top1 = rankings1[0]["index"] if rankings1 else None
    top2 = rankings2[0]["index"] if rankings2 else None
    top_rank_match = top1 == top2
    
    # Calculate score differences for common documents
    score_differences = []
    common_indices = set(scores1.keys()) & set(scores2.keys())
    
    for idx in common_indices:
        score1 = scores1[idx]
        score2 = scores2[idx]
        difference = abs(score1 - score2)
        
        score_differences.append({
            "index": idx,
            f"{model1_name}_score": score1,
            f"{model2_name}_score": score2,
            "difference": difference,
            "percentage_diff": (difference / score2 * 100) if score2 > 0 else 0
        })
    
    # Calculate average score difference
    avg_difference = sum(d["difference"] for d in score_differences) / len(score_differences) if score_differences else 0
    
    # Check if overall ranking order matches
    ranking_match = len(rankings1) == len(rankings2)
    if ranking_match:
        for i in range(len(rankings1)):
            if rankings1[i]["index"] != rankings2[i]["index"]:
                ranking_match = False
                break
    
    # Calculate correlation if we have scores
    correlation = 0.0
    if len(score_differences) > 1:
        scores1_list = [d[f"{model1_name}_score"] for d in score_differences]
        scores2_list = [d[f"{model2_name}_score"] for d in score_differences]
        correlation = np.corrcoef(scores1_list, scores2_list)[0, 1] if len(scores1_list) > 1 else 0.0
    
    return {
        "ranking_match": ranking_match,
        "top_rank_match": top_rank_match,
        "score_differences": score_differences,
        "avg_score_difference": avg_difference,
        "correlation": correlation,
        "top1_index": top1,
        "top2_index": top2
    }

def analyze_performance(results1: Dict, results2: Dict) -> Dict[str, Any]:
    """Analyze performance differences between two models"""
    times1 = []
    times2 = []
    
    for test_name in results1:
        if test_name in results2:
            result1 = results1[test_name]["result"]
            result2 = results2[test_name]["result"]
            
            if result1["success"] and result2["success"]:
                times1.append(result1["time"])
                times2.append(result2["time"])
    
    avg_time1 = sum(times1) / len(times1) if times1 else 0
    avg_time2 = sum(times2) / len(times2) if times2 else 0
    
    speed_ratio = avg_time2 / avg_time1 if avg_time1 > 0 else 0
    
    return {
        "avg_time1": avg_time1,
        "avg_time2": avg_time2,
        "speed_ratio": speed_ratio,
        "model1_faster": speed_ratio > 1
    }

def get_model_stats(results: Dict) -> Dict[str, Any]:
    """Get statistics for a single model"""
    total_tests = len(results)
    successful_tests = 0
    successful_times = []
    
    for r in results.values():
        if "result" in r and r["result"]["success"]:
            successful_tests += 1
            successful_times.append(r["result"]["time"])
    
    avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
    min_time = min(successful_times) if successful_times else 0
    max_time = max(successful_times) if successful_times else 0
    
    return {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time
    }

def compare_models_on_test_case(model1_results: Dict, model2_results: Dict, test_name: str, model1_name: str, model2_name: str) -> Dict[str, Any]:
    """Compare two models on a specific test case"""
    if test_name not in model1_results or test_name not in model2_results:
        return {}
    
    result1 = model1_results[test_name].get("result", {})
    result2 = model2_results[test_name].get("result", {})
    
    if not result1.get("success", False) or not result2.get("success", False):
        return {}
    
    comparison = compare_rankings(
        result1["results"], 
        result2["results"],
        model1_name, 
        model2_name
    )
    
    return {
        "test_name": test_name,
        "model1_time": result1["time"],
        "model2_time": result2["time"],
        "comparison": comparison
    }

def main():
    """Run comprehensive comparison analysis"""
    print("🔍 COMPREHENSIVE RERANKER RESULTS COMPARISON")
    print("=" * 70)
    
    # Load all result files
    result_files = glob.glob("results/*_results.json")
    all_results = {}
    
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace("_results.json", "")
        all_results[model_name] = load_results(file_path)
    
    if not all_results:
        print("❌ No result files found")
        return
    
    print(f"📊 Loaded {len(all_results)} model result files:")
    for model_name in all_results:
        print(f"  - {model_name}")
    
    # Group models by type
    bge_models = {k: v for k, v in all_results.items() if 'bge' in k}
    qwen_models = {k: v for k, v in all_results.items() if 'qwen' in k}
    
    print(f"\n📊 BGE Models: {len(bge_models)}")
    print(f"📊 Qwen Models: {len(qwen_models)}")
    
    # Individual model analysis
    print(f"\n📋 INDIVIDUAL MODEL ANALYSIS")
    print("=" * 70)
    
    model_stats = {}
    for model_name, results in all_results.items():
        stats = get_model_stats(results)
        model_stats[model_name] = stats
        
        print(f"\n🔍 {model_name}")
        print("-" * 40)
        print(f"  Total tests: {stats['total_tests']}")
        print(f"  Successful tests: {stats['successful_tests']}")
        print(f"  Success rate: {stats['success_rate']:.1f}%")
        print(f"  Average time: {stats['avg_time']:.3f}s")
        print(f"  Time range: {stats['min_time']:.3f}s - {stats['max_time']:.3f}s")
        
        # Show sample rankings for first successful test
        for test_name, test_result in results.items():
            if "result" in test_result and test_result["result"]["success"] and test_result["result"]["results"]:
                print(f"  Sample rankings from {test_name}:")
                for i, res in enumerate(test_result["result"]["results"][:3]):  # Show top 3
                    doc = res["document"]
                    score = res["relevance_score"]
                    print(f"    {i+1}. {doc[:50]}... (score: {score:.4f})")
                break
    
    # Compare BGE models
    if len(bge_models) > 1:
        print(f"\n🔍 BGE MODEL COMPARISONS")
        print("=" * 50)
        
        bge_model_names = list(bge_models.keys())
        for i in range(len(bge_model_names)):
            for j in range(i + 1, len(bge_model_names)):
                model1_name = bge_model_names[i]
                model2_name = bge_model_names[j]
                
                print(f"\n📊 Comparing {model1_name} vs {model2_name}")
                
                # Compare on all common test cases
                test_cases1 = set(bge_models[model1_name].keys())
                test_cases2 = set(bge_models[model2_name].keys())
                common_tests = test_cases1 & test_cases2
                
                if common_tests:
                    # Use first common test for detailed comparison
                    test_name = list(common_tests)[0]
                    comparison_result = compare_models_on_test_case(
                        bge_models[model1_name], 
                        bge_models[model2_name], 
                        test_name, 
                        model1_name, 
                        model2_name
                    )
                    
                    if comparison_result:
                        comp = comparison_result["comparison"]
                        print(f"  Test case: {test_name}")
                        print(f"  Top rank match: {'✅' if comp['top_rank_match'] else '❌'}")
                        print(f"  Full ranking match: {'✅' if comp['ranking_match'] else '❌'}")
                        print(f"  Avg score difference: {comp['avg_score_difference']:.4f}")
                        print(f"  Correlation: {comp['correlation']:.4f}")
                        print(f"  {model1_name} time: {comparison_result['model1_time']:.3f}s")
                        print(f"  {model2_name} time: {comparison_result['model2_time']:.3f}s")
    
    # Compare Qwen models
    if len(qwen_models) > 1:
        print(f"\n🔍 QWEN MODEL COMPARISONS")
        print("=" * 50)
        
        qwen_model_names = list(qwen_models.keys())
        for i in range(len(qwen_model_names)):
            for j in range(i + 1, len(qwen_model_names)):
                model1_name = qwen_model_names[i]
                model2_name = qwen_model_names[j]
                
                print(f"\n📊 Comparing {model1_name} vs {model2_name}")
                
                # Compare on all common test cases
                test_cases1 = set(qwen_models[model1_name].keys())
                test_cases2 = set(qwen_models[model2_name].keys())
                common_tests = test_cases1 & test_cases2
                
                if common_tests:
                    # Use first common test for detailed comparison
                    test_name = list(common_tests)[0]
                    comparison_result = compare_models_on_test_case(
                        qwen_models[model1_name], 
                        qwen_models[model2_name], 
                        test_name, 
                        model1_name, 
                        model2_name
                    )
                    
                    if comparison_result:
                        comp = comparison_result["comparison"]
                        print(f"  Test case: {test_name}")
                        print(f"  Top rank match: {'✅' if comp['top_rank_match'] else '❌'}")
                        print(f"  Full ranking match: {'✅' if comp['ranking_match'] else '❌'}")
                        print(f"  Avg score difference: {comp['avg_score_difference']:.4f}")
                        print(f"  Correlation: {comp['correlation']:.4f}")
                        print(f"  {model1_name} time: {comparison_result['model1_time']:.3f}s")
                        print(f"  {model2_name} time: {comparison_result['model2_time']:.3f}s")
    
    # Cross-model comparison (BGE vs Qwen)
    if bge_models and qwen_models:
        print(f"\n🔍 CROSS-MODEL COMPARISON (BGE vs Qwen)")
        print("=" * 60)
        
        # Compare first BGE model with first Qwen model
        bge_model_name = list(bge_models.keys())[0]
        qwen_model_name = list(qwen_models.keys())[0]
        
        print(f"📊 Comparing {bge_model_name} vs {qwen_model_name}")
        
        # Compare on all common test cases
        bge_test_cases = set(bge_models[bge_model_name].keys())
        qwen_test_cases = set(qwen_models[qwen_model_name].keys())
        common_tests = bge_test_cases & qwen_test_cases
        
        if common_tests:
            # Use first common test for detailed comparison
            test_name = list(common_tests)[0]
            comparison_result = compare_models_on_test_case(
                bge_models[bge_model_name], 
                qwen_models[qwen_model_name], 
                test_name, 
                bge_model_name, 
                qwen_model_name
            )
            
            if comparison_result:
                comp = comparison_result["comparison"]
                print(f"  Test case: {test_name}")
                print(f"  Top rank match: {'✅' if comp['top_rank_match'] else '❌'}")
                print(f"  Full ranking match: {'✅' if comp['ranking_match'] else '❌'}")
                print(f"  Avg score difference: {comp['avg_score_difference']:.4f}")
                print(f"  Correlation: {comp['correlation']:.4f}")
                print(f"  {bge_model_name} time: {comparison_result['model1_time']:.3f}s")
                print(f"  {qwen_model_name} time: {comparison_result['model2_time']:.3f}s")
    
    # Performance summary
    print(f"\n⚡ PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Sort by average time
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["avg_time"])
    
    print("Models ranked by average response time (fastest first):")
    for i, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"  {i}. {model_name}: {stats['avg_time']:.3f}s (min: {stats['min_time']:.3f}s, max: {stats['max_time']:.3f}s)")
    
    # Success rate summary
    print(f"\n📊 SUCCESS RATE SUMMARY")
    print("=" * 40)
    
    sorted_by_success = sorted(model_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True)
    for model_name, stats in sorted_by_success:
        print(f"  {model_name}: {stats['success_rate']:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
    
    # Save comprehensive comparison
    comparison_data = {
        "model_analysis": model_stats,
        "performance_ranking": sorted_models,
        "success_rate_ranking": sorted_by_success,
        "model_groups": {
            "bge_models": list(bge_models.keys()),
            "qwen_models": list(qwen_models.keys())
        }
    }
    
    with open("results/comprehensive_comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n💾 Comprehensive comparison saved to: results/comprehensive_comparison.json")

if __name__ == "__main__":
    main() 