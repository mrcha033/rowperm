#!/usr/bin/env python
"""Profile the impact of row permutation in a typical model training scenario.

This script measures what percentage of total training time is spent on
row permutation operations in a typical deep learning model.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim


class Timer:
    """Simple context manager for timing code blocks."""
    
    def __init__(self, name=None, verbose=True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose and self.name:
            print(f"{self.name}: {self.elapsed * 1000:.2f} ms")


class ModelWithPermutation(nn.Module):
    """Model with explicit row permutation operations."""
    
    def __init__(
        self, 
        input_dim=768, 
        hidden_dim=3072, 
        output_dim=768, 
        num_layers=4, 
        batch_size=16,
        seq_len=128,
        implementation="native",
        device="cuda"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.implementation = implementation
        self.device = device
        
        # Create layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Create fixed permutation indices for benchmarking
        self.indices = {}
        for i in range(num_layers):
            self.indices[i] = torch.randperm(seq_len, device=device)
            
        # Save timers for profiling
        self.timers = {
            "total": 0,
            "permutation": 0,
            "computation": 0,
        }
    
    def permute_rows(self, x, indices):
        """Permute rows using the specified implementation."""
        try:
            if self.implementation == "native":
                return x[indices]
            elif self.implementation == "cuda":
                import torch_rowperm
                if hasattr(torch_rowperm, "permute_rows_cuda"):
                    return torch_rowperm.permute_rows_cuda(x, indices)
                elif torch_rowperm.HAS_CUDA:
                    return torch_rowperm.permute_rows(x, indices)
                else:
                    return x[indices]
            elif self.implementation == "triton":
                import torch_rowperm
                if torch_rowperm.HAS_TRITON:
                    return torch_rowperm.permute_rows_triton(x, indices)
                else:
                    return x[indices]
            else:
                return x[indices]
        except Exception as e:
            print(f"Error in permute_rows: {e}")
            return x[indices]  # Fallback to native
    
    def forward(self, x):
        """Forward pass with timed permutation operations."""
        batch_size, seq_len, dim = x.shape
        total_timer = Timer("total", verbose=False)
        perm_timer = Timer("permutation", verbose=False)
        comp_timer = Timer("computation", verbose=False)
        
        with total_timer:
            for i, layer in enumerate(self.layers):
                # Process through layer
                with comp_timer:
                    x = layer(x)
                
                # Apply permutation between layers (except last)
                if i < len(self.layers) - 1:
                    indices = self.indices[i]
                    with perm_timer:
                        # Apply permutation to each sequence in batch
                        for b in range(batch_size):
                            x[b] = self.permute_rows(x[b], indices)
        
        # Update timers
        self.timers["total"] += total_timer.elapsed
        self.timers["permutation"] += perm_timer.elapsed
        self.timers["computation"] += comp_timer.elapsed
        
        return x


def profile_model(
    batch_size=16,
    seq_len=128, 
    input_dim=768,
    hidden_dim=3072,
    num_layers=4,
    num_iterations=10,
    implementation="native",
    device="cuda"
):
    """Profile a model with row permutation operations."""
    # Create model
    model = ModelWithPermutation(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim,
        num_layers=num_layers,
        batch_size=batch_size,
        seq_len=seq_len,
        implementation=implementation,
        device=device
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Reset timers
    model.timers = {
        "total": 0,
        "permutation": 0,
        "computation": 0,
    }
    
    # Run iterations
    for i in range(num_iterations):
        # Generate random input and target
        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        target = torch.randn(batch_size, seq_len, input_dim, device=device)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (i + 1) % 5 == 0:
            print(f"Iteration {i + 1}/{num_iterations} complete")
    
    # Calculate average times and percentages
    avg_times = {}
    for key, value in model.timers.items():
        avg_times[key] = (value / num_iterations) * 1000  # Convert to ms
    
    # Calculate percentages
    percentages = {}
    total_time = avg_times["total"]
    for key in ["permutation", "computation"]:
        percentages[key] = (avg_times[key] / total_time) * 100
    
    return avg_times, percentages


def get_available_implementations():
    """Get list of available implementations."""
    implementations = ["native"]
    
    try:
        import torch_rowperm
        if torch_rowperm.HAS_CUDA:
            implementations.append("cuda")
        if torch_rowperm.HAS_TRITON:
            implementations.append("triton")
    except ImportError:
        pass
    
    return implementations


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Profile row permutation in models")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--input-dim", type=int, default=768, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=3072, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-iterations", type=int, default=20, help="Number of iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    # Get available implementations
    implementations = get_available_implementations()
    parser.add_argument("--impl", choices=implementations, default="native",
                       help="Permutation implementation")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Print configuration
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Implementation: {args.impl}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Input dimension: {args.input_dim}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Iterations: {args.num_iterations}")
    
    # Run profiling
    print("\nRunning profiling...")
    avg_times, percentages = profile_model(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_iterations=args.num_iterations,
        implementation=args.impl,
        device=args.device
    )
    
    # Print results
    print("\nResults:")
    print(f"  Total time per iteration: {avg_times['total']:.2f} ms")
    print(f"  Permutation time: {avg_times['permutation']:.2f} ms ({percentages['permutation']:.2f}%)")
    print(f"  Computation time: {avg_times['computation']:.2f} ms ({percentages['computation']:.2f}%)")


if __name__ == "__main__":
    main() 