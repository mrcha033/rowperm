#!/usr/bin/env python
"""Profile the impact of row permutation in a typical model training scenario."""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Try to import torch_rowperm, with fallbacks
try:
    import torch_rowperm
    HAS_ROWPERM = True
    
    if torch_rowperm.HAS_TRITON:
        IMPLEMENTATIONS = ["native", "cuda", "triton"]
    elif torch_rowperm.HAS_CUDA:
        IMPLEMENTATIONS = ["native", "cuda"]
    else:
        IMPLEMENTATIONS = ["native"]
except ImportError:
    HAS_ROWPERM = False
    IMPLEMENTATIONS = ["native"]


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name: str = None):
        self.name = name
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed * 1000:.2f} ms")


class SimpleTransformerWithPermutation(nn.Module):
    """Simple transformer-like model with row permutation."""
    
    def __init__(
        self, 
        vocab_size: int = 10000, 
        hidden_size: int = 768, 
        num_layers: int = 4,
        perm_implementation: str = "native"
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=8, 
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification
        self.perm_implementation = perm_implementation
        
    def permute_rows(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Permute rows using the specified implementation."""
        if self.perm_implementation == "native":
            return x[indices]
        elif self.perm_implementation == "cuda" and HAS_ROWPERM and torch_rowperm.HAS_CUDA:
            if hasattr(torch_rowperm, "permute_rows_cuda"):
                return torch_rowperm.permute_rows_cuda(x, indices)
            return torch_rowperm.permute_rows(x, indices)
        elif self.perm_implementation == "triton" and HAS_ROWPERM and torch_rowperm.HAS_TRITON:
            return torch_rowperm.permute_rows_triton(x, indices)
        else:
            return x[indices]  # Fallback
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with permutation."""
        # Standard embedding
        x = self.embedding(x)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Permute rows between layers to simulate reordering
            # This simulates shuffling token order in the sequence
            if i < len(self.layers) - 1:  # Skip permutation after last layer
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                # Create permutation indices for each sequence in the batch
                indices = torch.stack([
                    torch.randperm(seq_len, device=x.device)
                    for _ in range(batch_size)
                ])
                
                # Permute each sequence separately
                with Timer(f"permute_layer_{i}"):
                    for b in range(batch_size):
                        x[b] = self.permute_rows(x[b], indices[b])
        
        # Average pooling and classification
        x = torch.mean(x, dim=1)
        return self.classifier(x)


def run_benchmark(
    implementation: str,
    batch_size: int = 16,
    seq_len: int = 128,
    hidden_size: int = 768,
    num_layers: int = 4,
    num_iterations: int = 5,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run benchmark with specified parameters."""
    # Create model
    model = SimpleTransformerWithPermutation(
        vocab_size=10000,
        hidden_size=hidden_size,
        num_layers=num_layers,
        perm_implementation=implementation
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Timing accumulators
    times = {
        "total": 0,
        "forward": 0,
        "backward": 0,
        "permutation": 0
    }
    
    # Run iterations
    for i in range(num_iterations):
        # Create random input data
        inputs = torch.randint(0, 10000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device)
        
        # Reset permutation timers
        for j in range(num_layers - 1):
            times[f"permute_layer_{j}"] = 0
            
        # Full iteration timing
        with Timer() as iteration_timer:
            # Forward pass
            with Timer() as forward_timer:
                outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            with Timer() as backward_timer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Accumulate times
        times["total"] += iteration_timer.elapsed
        times["forward"] += forward_timer.elapsed
        
        # Extract permutation times
        for j in range(num_layers - 1):
            layer_perm_time = times.get(f"permute_layer_{j}", 0)
            times["permutation"] += layer_perm_time
            
        times["backward"] += backward_timer.elapsed
        
    # Average times
    for key in times:
        times[key] = (times[key] / num_iterations) * 1000  # Convert to ms
        
    # Calculate percentages
    total_ms = times["total"]
    for key in ["forward", "backward", "permutation"]:
        times[f"{key}_pct"] = (times[key] / total_ms) * 100 if total_ms > 0 else 0
    
    return times


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Profile row permutation impact")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--implementations", nargs="+", choices=IMPLEMENTATIONS, default=IMPLEMENTATIONS,
                      help="Implementations to benchmark")
    args = parser.parse_args()
    
    # Check if we have a GPU
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Device: {args.device}")
    print(f"  Implementations: {args.implementations}")
    
    # Print availability
    if HAS_ROWPERM:
        print(f"torch_rowperm available:")
        print(f"  CUDA: {torch_rowperm.HAS_CUDA}")
        print(f"  Triton: {torch_rowperm.HAS_TRITON}")
    else:
        print("torch_rowperm not available")
    
    # Warmup
    print("\nWarming up...")
    _ = run_benchmark(
        implementation="native",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_iterations=2,
        device=args.device
    )
    
    # Run benchmarks
    results = {}
    print("\nRunning benchmarks...")
    for impl in args.implementations:
        print(f"\nImplementation: {impl}")
        results[impl] = run_benchmark(
            implementation=impl,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_iterations=args.num_iterations,
            device=args.device
        )
    
    # Print summary
    print("\nResults:")
    print(f"{'Implementation':<10} {'Total (ms)':<12} {'Permutation (ms)':<18} {'Permutation %':<15}")
    print("-" * 55)
    for impl, times in results.items():
        print(f"{impl:<10} {times['total']:<12.2f} {times['permutation']:<18.2f} {times['permutation_pct']:<15.2f}")
    
    # If we have multiple implementations, print comparison
    if len(results) > 1 and "native" in results:
        base_time = results["native"]["permutation"]
        print("\nSpeedup vs Native:")
        for impl, times in results.items():
            if impl != "native":
                speedup = base_time / times["permutation"] if times["permutation"] > 0 else float('inf')
                print(f"  {impl}: {speedup:.2f}x")
    
    # Print wall-time impact
    for impl, times in results.items():
        print(f"\n{impl} permutation impact on total training time: {times['permutation_pct']:.2f}%")


if __name__ == "__main__":
    main()
