#!/usr/bin/env python
"""Benchmark and profile row permutation operations.

This script measures:
1. Speedup of Triton vs native PyTorch for various tensor sizes
2. Wall-time impact of row permutation in model training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn


def run_benchmark():
    """Run benchmark comparing native PyTorch vs Triton."""
    print("\n" + "="*80)
    print("BENCHMARKING ROW PERMUTATION OPERATIONS")
    print("="*80)
    
    try:
        from torch_rowperm._triton.row_perm import benchmark_vs_native
    except ImportError:
        print("Error: torch_rowperm._triton.row_perm not found.")
        return
    
    # Define benchmark sizes - various tensor shapes to test different patterns
    sizes = [
        (1000, 128),     # Small
        (10000, 256),    # Medium
        (50000, 512),    # Large
        (100000, 768),   # Very large
        (5000, 3072),    # Wide rows
        (20000, 1024),   # Common in transformers
    ]
    
    # Run the benchmark
    results = benchmark_vs_native(
        sizes=sizes,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        iterations=10
    )
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    avg_speedup = sum(r['speedup'] for r in results.values()) / len(results)
    print(f"Average speedup across all sizes: {avg_speedup:.2f}x")
    
    max_speedup = max(r['speedup'] for r in results.values())
    max_size = [k for k, v in results.items() if v['speedup'] == max_speedup][0]
    print(f"Best speedup: {max_speedup:.2f}x for size {max_size}")
    
    return results


class SimpleModelWithPermutation(nn.Module):
    """Simple model with row permutation operations for profiling."""
    
    def __init__(
        self, 
        batch_size=8,
        seq_len=128,
        hidden_dim=768,
        num_layers=4,
        implementation='native',
        device='cuda'
    ):
        super().__init__()
        self.implementation = implementation
        self.device = device
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Create indices for permutation - fixed for reproducible benchmarking
        self.indices = [torch.randperm(seq_len, device=device) for _ in range(num_layers)]
        
        # Initialize timer
        self.permutation_time = 0
        self.forward_time = 0
    
    def permute_rows(self, x, indices):
        """Permute rows using specified implementation."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        if self.implementation == 'native':
            result = x[indices]
        elif self.implementation == 'triton':
            from torch_rowperm._triton.row_perm import permute_rows_triton
            result = permute_rows_triton(x, indices)
        else:
            result = x[indices]  # Fallback to native
            
        end.record()
        torch.cuda.synchronize()
        self.permutation_time += start.elapsed_time(end) / 1000  # Convert ms to s
        
        return result
    
    def forward(self, x):
        """Forward pass with permutation between layers."""
        batch_size, seq_len, hidden_dim = x.shape
        
        for i, layer in enumerate(self.layers):
            # Apply layer
            x = layer(x)
            
            # Apply permutation between layers (except last)
            if i < len(self.layers) - 1:
                indices = self.indices[i]
                # Apply permutation to each sequence in batch
                for b in range(batch_size):
                    x[b] = self.permute_rows(x[b], indices)
        
        return x


def run_profiling():
    """Profile the impact of row permutation in model training."""
    print("\n" + "="*80)
    print("PROFILING ROW PERMUTATION IMPACT")
    print("="*80)
    
    try:
        from torch_rowperm._triton.row_perm import profile_permutation_impact
    except ImportError:
        print("Error: torch_rowperm._triton.row_perm not found.")
        return
    
    # Check available implementations
    implementations = ['native']
    try:
        import triton
        implementations.append('triton')
    except ImportError:
        pass
    
    # Run profiling for different model sizes
    model_sizes = [
        # (batch_size, seq_len, hidden_dim, name)
        (16, 128, 768, "BERT-base sized"),
        (8, 512, 1024, "GPT-2 sized"),
        (4, 2048, 2560, "LLaMa-7B sized")
    ]
    
    all_results = {}
    
    for batch, seq_len, hidden_dim, name in model_sizes:
        print(f"\nProfiling {name} model (batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim})")
        
        results = profile_permutation_impact(
            model_func=SimpleModelWithPermutation,
            batch_size=batch,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            implementations=implementations
        )
        
        all_results[name] = results
    
    # Print summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY - PERMUTATION IMPACT ON WALL TIME")
    print("="*80)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name} model:")
        for impl, stats in results.items():
            pct = stats.get('permutation_percentage', 0)
            print(f"  {impl}: {pct:.2f}% of total wall time")
    
    return all_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark row permutation operations")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--profile", action="store_true", help="Run profiling")
    parser.add_argument("--all", action="store_true", help="Run both benchmark and profiling")
    args = parser.parse_args()
    
    # Default to all if nothing specified
    if not (args.benchmark or args.profile or args.all):
        args.all = True
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
    
    # Run benchmark
    benchmark_results = None
    if args.benchmark or args.all:
        benchmark_results = run_benchmark()
    
    # Run profiling
    profile_results = None
    if args.profile or args.all:
        profile_results = run_profiling()
    
    # Print final ROI assessment
    if profile_results:
        print("\n" + "="*80)
        print("ROI ASSESSMENT")
        print("="*80)
        
        max_pct = 0
        for model, results in profile_results.items():
            for impl, stats in results.items():
                pct = stats.get('permutation_percentage', 0)
                max_pct = max(max_pct, pct)
        
        if max_pct < 3:
            print("ROI ASSESSMENT: LOW (< 3% of wall time)")
            print("Recommendation: Current optimization level is sufficient")
        elif max_pct < 10:
            print("ROI ASSESSMENT: MEDIUM (3-10% of wall time)")
            print("Recommendation: Consider targeted optimizations for specific model sizes")
        else:
            print("ROI ASSESSMENT: HIGH (>10% of wall time)")
            print("Recommendation: Further optimization would be valuable")


if __name__ == "__main__":
    main() 