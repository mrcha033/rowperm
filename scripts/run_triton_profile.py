#!/usr/bin/env python
"""Run profiling for different model sizes and implementations."""

import os
import sys
import subprocess
import argparse


def run_profiling(model_sizes, implementations):
    """Run profiling for different model sizes and implementations."""
    for impl in implementations:
        print(f"\n\n{'='*80}")
        print(f"RUNNING PROFILER WITH IMPLEMENTATION: {impl}")
        print(f"{'='*80}\n")
        
        for size in model_sizes:
            batch_size, seq_len, hidden_dim = size
            
            print(f"\n{'-'*50}")
            print(f"MODEL SIZE: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")
            print(f"{'-'*50}\n")
            
            cmd = [
                "python", "scripts/profile_model_impact.py",
                "--batch-size", str(batch_size),
                "--seq-len", str(seq_len),
                "--hidden-dim", str(hidden_dim),
                "--num-iterations", "10",
                "--impl", impl
            ]
            
            subprocess.run(cmd)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run profiling for different model sizes")
    parser.add_argument("--small", action="store_true", help="Run small model only")
    parser.add_argument("--medium", action="store_true", help="Run medium model only")
    parser.add_argument("--large", action="store_true", help="Run large model only")
    parser.add_argument("--all", action="store_true", help="Run all model sizes")
    parser.add_argument("--native", action="store_true", help="Run native PyTorch implementation")
    parser.add_argument("--cuda", action="store_true", help="Run CUDA implementation")
    parser.add_argument("--triton", action="store_true", help="Run Triton implementation")
    args = parser.parse_args()
    
    # Default to all if no specific size is selected
    if not (args.small or args.medium or args.large or args.all):
        args.medium = True
    
    # Default to all implementations if none specified
    if not (args.native or args.cuda or args.triton):
        args.native = True
        args.cuda = True
        args.triton = True
    
    # Define model sizes
    model_sizes = []
    
    # Small model (like BERT-base)
    if args.small or args.all:
        model_sizes.append((16, 128, 768))
    
    # Medium model (like BERT-large or GPT-2)
    if args.medium or args.all:
        model_sizes.append((8, 512, 1024))
    
    # Large model (like GPT-3 or LLaMa)
    if args.large or args.all:
        model_sizes.append((4, 2048, 4096))
    
    # Define implementations
    impls = []
    if args.native:
        impls.append("native")
    if args.cuda:
        impls.append("cuda")
    if args.triton:
        impls.append("triton")
    
    # Run profiling
    run_profiling(model_sizes, impls)


if __name__ == "__main__":
    main() 