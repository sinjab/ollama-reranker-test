#!/usr/bin/env python3
"""
Compare Ollama vs Official BGE Reranker Results
==============================================

Compares the latest Ollama results with official BGE reranker results
to analyze performance differences, ranking accuracy, and score distributions.
"""

import json
import os
from typing import Dict, List, Any

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return {}

def compare_rankings(ollama_rankings: List[Dict], official_rankings: List[Dict]) -> Dict[str, Any]:
    """Compare rankings between Ollama and official results"""
    if not ollama_rankings or not official_rankings:
        return {
            "ranking_match": False,
            "score_differences": [],
            "top_rank_match": False,
            "correlation": 0.0
        }
    
    # Create index to score mappings
    ollama_scores = {r["index"]: r["relevance_score"] for r in ollama_rankings}
    official_scores = {r["index"]: r["relevance_score"] for r in official_rankings}
    
    # Check if top rank matches
    ollama_top = ollama_rankings[0]["index"] if ollama_rankings else None
    official_top = official_rankings[0]["index"] if official_rankings else None
    top_rank_match = ollama_top == official_top
    
    # Calculate score differences for common documents
    score_differences = []
    common_indices = set(ollama_scores.keys()) & set(official_scores.keys())
    
    for idx in common_indices:
        ollama_score = ollama_scores[idx]
        official_score = official_scores[idx]
        difference = abs(ollama_score - official_score)
        
        score_differences.append({
            "index": idx,
            "ollama_score": ollama_score,
            "official_score": official_score,
            "difference": difference,
            "percentage_diff": (difference / official_score * 100) if official_score > 0 else 0
        })
    
    # Calculate average score difference
    avg_difference = sum(d["difference"] for d in score_differences) / len(score_differences) if score_differences else 0
    
    # Check if overall ranking order matches (simplified)
    ranking_match = len(ollama_rankings) == len(official_rankings)
    if ranking_match:
        for i in range(len(ollama_rankings)):
            if ollama_rankings[i]["index"] != official_rankings[i]["index"]:
                ranking_match = False
                break
    
    return {
        "ranking_match": ranking_match,
        "top_rank_match": top_rank_match,
        "score_differences": score_differences,
        "avg_score_difference": avg_difference,
        "ollama_top_index": ollama_top,
        "official_top_index": official_top
    }

def analyze_performance(ollama_results: Dict, official_results: Dict) -> Dict[str, Any]:
    """Analyze performance differences"""
    ollama_times = []
    official_times = []
    
    for test_name in ollama_results:
        if test_name in official_results:
            ollama_result = ollama_results[test_name]["result"]
            official_result = official_results[test_name]["result"]
            
            if ollama_result["success"] and official_result["success"]:
                ollama_times.append(ollama_result["time"])
                official_times.append(official_result["time"])
    
    avg_ollama_time = sum(ollama_times) / len(ollama_times) if ollama_times else 0
    avg_official_time = sum(official_times) / len(official_times) if official_times else 0
    
    speed_ratio = avg_official_time / avg_ollama_time if avg_ollama_time > 0 else 0
    
    return {
        "avg_ollama_time": avg_ollama_time,
        "avg_official_time": avg_official_time,
        "speed_ratio": speed_ratio,
        "ollama_faster": speed_ratio > 1
    }

def main():
    """Run comparison analysis"""
    print("🔍 Comparing Ollama vs Official BGE Reranker Results")
    print("=" * 60)
    
    # Load results
    ollama_results = load_results("results/ollama_results.json")
    official_results = load_results("results/bge_official_results.json")
    
    if not ollama_results or not official_results:
        print("❌ Could not load results files")
        return
    
    print(f"📊 Loaded {len(ollama_results)} Ollama tests")
    print(f"📊 Loaded {len(official_results)} Official tests")
    
    # Find the official model results (first model in the dict)
    official_model_name = list(official_results.keys())[0] if official_results else None
    if not official_model_name:
        print("❌ No official model results found")
        return
    
    official_model_results = official_results[official_model_name]
    print(f"📊 Using official model: {official_model_name}")
    
    # Compare each test case
    comparison_results = {}
    total_tests = 0
    successful_comparisons = 0
    ranking_matches = 0
    top_rank_matches = 0
    
    for test_name in ollama_results:
        if test_name in official_model_results:
            total_tests += 1
            
            ollama_data = ollama_results[test_name]
            official_data = official_model_results[test_name]
            
            # Skip if either test failed
            if not ollama_data["result"]["success"] or not official_data["result"]["success"]:
                continue
            
            successful_comparisons += 1
            
            # Compare rankings
            comparison = compare_rankings(
                ollama_data["result"]["results"],
                official_data["result"]["results"]
            )
            
            if comparison["ranking_match"]:
                ranking_matches += 1
            if comparison["top_rank_match"]:
                top_rank_matches += 1
            
            comparison_results[test_name] = {
                "test_name": test_name,
                "ollama_success": ollama_data["result"]["success"],
                "official_success": official_data["result"]["success"],
                "ollama_rankings": ollama_data["result"]["results"],
                "official_rankings": official_data["result"]["results"],
                "ollama_time": ollama_data["result"]["time"],
                "official_time": official_data["result"]["time"],
                **comparison
            }
    
    # Performance analysis
    performance = analyze_performance(ollama_results, official_model_results)
    
    # Print summary
    print(f"\n📈 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Total comparable tests: {total_tests}")
    print(f"Successful comparisons: {successful_comparisons}")
    
    if successful_comparisons > 0:
        print(f"Ranking matches: {ranking_matches}/{successful_comparisons} ({ranking_matches/successful_comparisons*100:.1f}%)")
        print(f"Top rank matches: {top_rank_matches}/{successful_comparisons} ({top_rank_matches/successful_comparisons*100:.1f}%)")
    else:
        print("No successful comparisons to analyze")
    
    print(f"\n⚡ PERFORMANCE")
    print(f"Average Ollama time: {performance['avg_ollama_time']:.3f}s")
    print(f"Average Official time: {performance['avg_official_time']:.3f}s")
    print(f"Speed ratio: {performance['speed_ratio']:.2f}x")
    print(f"Ollama is {'faster' if performance['ollama_faster'] else 'slower'}")
    
    # Detailed results for each test
    print(f"\n📋 DETAILED RESULTS")
    print("=" * 60)
    
    for test_name, comparison in comparison_results.items():
        print(f"\n🔍 {test_name}")
        print(f"  Top rank match: {'✅' if comparison['top_rank_match'] else '❌'}")
        print(f"  Full ranking match: {'✅' if comparison['ranking_match'] else '❌'}")
        avg_diff = comparison.get('avg_score_difference', 0)
        print(f"  Avg score difference: {avg_diff:.4f}")
        print(f"  Ollama time: {comparison['ollama_time']:.3f}s")
        print(f"  Official time: {comparison['official_time']:.3f}s")
        
        # Show top rankings
        if comparison['ollama_rankings'] and comparison['official_rankings']:
            print("  Top rankings:")
            ollama_top = comparison['ollama_rankings'][0]
            official_top = comparison['official_rankings'][0]
            print(f"    Ollama: {ollama_top['document'][:50]}... (score: {ollama_top['relevance_score']:.4f})")
            print(f"    Official: {official_top['document'][:50]}... (score: {official_top['relevance_score']:.4f})")
    
    # Save comparison results
    output = {
        "summary": {
            "total_tests": total_tests,
            "successful_comparisons": successful_comparisons,
            "ranking_matches": ranking_matches,
            "top_rank_matches": top_rank_matches,
            "ranking_match_rate": ranking_matches/successful_comparisons if successful_comparisons > 0 else 0,
            "top_rank_match_rate": top_rank_matches/successful_comparisons if successful_comparisons > 0 else 0
        },
        "performance": performance,
        "test_cases": comparison_results
    }
    
    # Save to file
    with open("results/latest_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Detailed comparison saved to: results/latest_comparison.json")

if __name__ == "__main__":
    main() 