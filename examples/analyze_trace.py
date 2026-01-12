#!/usr/bin/env python3
"""
Offline Trace Analysis Script for SP Scheduling Hyperparameter Tuning

This script analyzes trace_data.jsonl to derive optimal hyperparameters:
- learned_long_req_threshold: Multiplier for avg_prompt_length to identify "long" requests
- learned_imbalance_threshold: KVCache imbalance ratio threshold for triggering SP

Usage:
    python analyze_trace.py trace_data.jsonl --output optimal_hyperparams.json
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
import statistics


def load_trace_data(filepath: str) -> List[Dict[str, Any]]:
    """Load trace data from JSONL file (supports both single-line and multi-line JSON)."""
    traces = []
    current_json = []
    brace_count = 0
    line_num = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line_num += 1
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Track braces to detect complete JSON objects
            brace_count += line_stripped.count('{') - line_stripped.count('}')
            current_json.append(line)
            
            # If braces are balanced, we have a complete JSON object
            if brace_count == 0 and current_json:
                try:
                    json_str = ''.join(current_json).strip()
                    trace = json.loads(json_str)
                    traces.append(trace)
                    current_json = []
                except json.JSONDecodeError as e:
                    # Try to find where the JSON object actually ends
                    # Sometimes there might be trailing content
                    json_str = ''.join(current_json).strip()
                    # Try to extract just the JSON part
                    try:
                        # Find the last closing brace
                        last_brace = json_str.rfind('}')
                        if last_brace > 0:
                            json_str = json_str[:last_brace + 1]
                            trace = json.loads(json_str)
                            traces.append(trace)
                            current_json = []
                        else:
                            print(f"Warning: Failed to parse JSON starting at line {line_num - len(current_json) + 1}: {e}")
                            current_json = []
                    except:
                        print(f"Warning: Failed to parse JSON starting at line {line_num - len(current_json) + 1}: {e}")
                        current_json = []
    
    # Handle any remaining incomplete JSON
    if current_json:
        try:
            json_str = ''.join(current_json).strip()
            trace = json.loads(json_str)
            traces.append(trace)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse incomplete JSON at end of file: {e}")
    
    print(f"Loaded {len(traces)} trace samples from {filepath}")
    return traces


def analyze_long_request_threshold(traces: List[Dict]) -> float:
    """
    Derive optimal learned_long_req_threshold.
    
    Strategy:
    1. Calculate avg_prompt_length from all traces
    2. Identify requests that cause batch_size_cv > 0.3 (communication imbalance)
    3. Find the threshold multiplier that best separates "problematic long requests"
       from normal requests
    """
    if not traces:
        return 3.0  # Default
    
    # Calculate average prompt length
    prompt_lengths = [t['prompt_length'] for t in traces]
    avg_prompt_length = statistics.mean(prompt_lengths)
    
    # Identify "pain points" where batch_size_cv is high
    pain_threshold_cv = 0.3
    pain_points = [t for t in traces if t.get('batch_size_cv', 0) > pain_threshold_cv]
    
    if not pain_points:
        print("  No pain points found (batch_size_cv > 0.3), using default threshold")
        return 3.0
    
    # Analyze prompt lengths at pain points
    pain_prompt_lengths = [t['prompt_length'] for t in pain_points]
    pain_avg_prompt = statistics.mean(pain_prompt_lengths)
    
    # Calculate threshold as ratio
    threshold_multiplier = pain_avg_prompt / avg_prompt_length if avg_prompt_length > 0 else 3.0
    
    # Clamp to reasonable range [1.5, 5.0]
    threshold_multiplier = max(1.5, min(5.0, threshold_multiplier))
    
    print(f"  Average prompt length: {avg_prompt_length:.2f}")
    print(f"  Pain points (batch_size_cv > {pain_threshold_cv}): {len(pain_points)}/{len(traces)}")
    print(f"  Average prompt length at pain points: {pain_avg_prompt:.2f}")
    print(f"  Derived threshold multiplier: {threshold_multiplier:.3f}")
    
    return threshold_multiplier


def analyze_imbalance_threshold(traces: List[Dict]) -> float:
    """
    Derive optimal learned_imbalance_threshold.
    
    Strategy:
    1. Identify moments with high kvcache_imbalance_ratio
    2. Check if these moments also have high batch_size_cv (indicating persistent imbalance)
    3. Find the threshold that best separates "persistent imbalance" from transient spikes
    """
    if not traces:
        return 1.5  # Default
    
    # Get all imbalance ratios
    imbalance_ratios = [t.get('kvcache_imbalance_ratio', 1.0) for t in traces]
    
    # Identify persistent imbalance: high imbalance AND high batch_size_cv
    # This indicates a sustained problem that DP routing alone cannot solve
    persistent_imbalance_threshold_cv = 0.25
    persistent_imbalance_points = [
        t for t in traces 
        if t.get('kvcache_imbalance_ratio', 1.0) > 1.2 and 
           t.get('batch_size_cv', 0) > persistent_imbalance_threshold_cv
    ]
    
    if not persistent_imbalance_points:
        # Fallback: use 75th percentile of imbalance ratios
        threshold = np.percentile(imbalance_ratios, 75)
        print("  No persistent imbalance points found, using 75th percentile")
    else:
        # Use median imbalance ratio at persistent imbalance points
        persistent_imbalances = [t.get('kvcache_imbalance_ratio', 1.0) 
                               for t in persistent_imbalance_points]
        threshold = statistics.median(persistent_imbalances)
        print(f"  Persistent imbalance points: {len(persistent_imbalance_points)}/{len(traces)}")
    
    # Clamp to reasonable range [1.1, 2.5]
    threshold = max(1.1, min(2.5, threshold))
    
    print(f"  Derived imbalance threshold: {threshold:.3f}")
    
    return threshold


def analyze_statistics(traces: List[Dict]) -> Dict[str, float]:
    """Extract average statistics from traces."""
    if not traces:
        return {}
    
    # Separate short and long requests (using a heuristic: < 2x avg as short)
    prompt_lengths = [t['prompt_length'] for t in traces]
    avg_prompt = statistics.mean(prompt_lengths)
    
    short_traces = [t for t in traces if t['prompt_length'] < 2.0 * avg_prompt]
    
    stats = {
        'avg_prompt_length': avg_prompt,
        'avg_output_length': statistics.mean([t.get('avg_short_output_length', 256) for t in traces]),
    }
    
    if short_traces:
        stats['avg_short_prompt_length'] = statistics.mean([t['prompt_length'] for t in short_traces])
        stats['avg_short_output_length'] = statistics.mean([
            t.get('avg_short_output_length', 256) for t in short_traces
        ])
        # Estimate avg_short_batch_size from batch_size_per_rank
        all_short_batch_sizes = []
        for t in short_traces:
            batch_sizes = t.get('batch_size_per_rank', [])
            if batch_sizes:
                all_short_batch_sizes.extend(batch_sizes)
        if all_short_batch_sizes:
            stats['avg_short_batch_size'] = statistics.mean(all_short_batch_sizes)
    
    return stats


def derive_hyperparameters(traces: List[Dict]) -> Dict[str, Any]:
    """Main analysis function to derive optimal hyperparameters."""
    print("=" * 60)
    print("Analyzing Trace Data for Hyperparameter Tuning")
    print("=" * 60)
    
    if not traces:
        print("Error: No trace data loaded!")
        return {}
    
    # Derive thresholds
    print("\n[1] Analyzing Long Request Threshold...")
    long_req_threshold = analyze_long_request_threshold(traces)
    
    print("\n[2] Analyzing Imbalance Threshold...")
    imbalance_threshold = analyze_imbalance_threshold(traces)
    
    # Extract statistics
    print("\n[3] Extracting Statistics...")
    stats = analyze_statistics(traces)
    
    # Build hyperparameters dict
    hyperparams = {
        'learned_long_req_threshold': round(long_req_threshold, 3),
        'learned_imbalance_threshold': round(imbalance_threshold, 3),
        'avg_short_prompt_length': round(stats.get('avg_short_prompt_length', stats.get('avg_prompt_length', 1024)), 2),
        'avg_short_output_length': round(stats.get('avg_short_output_length', 256), 2),
        'avg_short_batch_size': round(stats.get('avg_short_batch_size', 50), 2),
        'avg_prompt_length': round(stats.get('avg_prompt_length', 1024), 2),
        'avg_output_length': round(stats.get('avg_output_length', 256), 2),
        'sp_decision_count': 0,  # Will be updated during runtime
        'running_benefit_ratio': 0.5  # Default, will be learned online
    }
    
    print("\n" + "=" * 60)
    print("Derived Hyperparameters:")
    print("=" * 60)
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    return hyperparams


def main():
    parser = argparse.ArgumentParser(
        description='Analyze trace data to derive optimal SP scheduling hyperparameters'
    )
    parser.add_argument('trace_file', type=str, help='Path to trace_data.jsonl file')
    parser.add_argument('--output', '-o', type=str, default='optimal_hyperparams.json',
                       help='Output file path for hyperparameters (default: optimal_hyperparams.json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed analysis')
    
    args = parser.parse_args()
    
    # Load trace data
    traces = load_trace_data(args.trace_file)
    
    if not traces:
        print(f"Error: No valid trace data found in {args.trace_file}")
        return 1
    
    # Derive hyperparameters
    hyperparams = derive_hyperparameters(traces)
    
    if not hyperparams:
        print("Error: Failed to derive hyperparameters")
        return 1
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print(f"\n✓ Hyperparameters saved to: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review the hyperparameters in {args.output}")
    print(f"  2. Use --load-hyperparams {args.output} when running with SP enabled")
    print(f"  3. Monitor performance and adjust if needed")
    
    return 0


if __name__ == '__main__':
    exit(main())
