import numpy as np
import time
import numba
from numba import cuda
import dask.array as da
import os
import argparse

# -------------------------------------------------------------------------
# 1. CPU Implementation (NumPy Vectorized)
# -------------------------------------------------------------------------
def cpu_complement(data):
    """
    Computes the DNA complement using pure NumPy.
    Since A=0, C=1, G=2, T=3, complement is simply 3 - data.
    """
    return 3 - data

# -------------------------------------------------------------------------
# 2. Numba CPU Implementation (JIT Compiled)
# -------------------------------------------------------------------------
@numba.jit(nopython=True)
def numba_cpu_complement(data):
    """
    Computes the DNA complement using Numba JIT (CPU).
    """
    res = np.empty_like(data)
    for i in range(data.size):
        res[i] = 3 - data[i]
    return res

# -------------------------------------------------------------------------
# 3. Numba CUDA Implementation (GPU)
# -------------------------------------------------------------------------
@cuda.jit
def numba_gpu_complement_kernel(data, res):
    """
    CUDA Kernel to compute the DNA complement.
    Each thread processes one nucleotide base.
    """
    i = cuda.grid(1)
    if i < data.size:
        res[i] = 3 - data[i]

def numba_gpu_complement(data):
    """
    Wrapper function to execute the CUDA kernel.
    Handles host-to-device memory transfer and kernel launch.
    """
    # Allocate device memory and copy data from host (CPU) to device (GPU)
    d_data = cuda.to_device(data)
    d_res = cuda.device_array_like(data)
    
    # Configure grid and block dimensions
    threadsperblock = 256
    blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    
    # Launch kernel
    numba_gpu_complement_kernel[blockspergrid, threadsperblock](d_data, d_res)
    
    # Wait for the GPU to finish execution
    cuda.synchronize()
    
    # Copy the result back from device (GPU) to host (CPU)
    return d_res.copy_to_host()

# -------------------------------------------------------------------------
# 4. Dask Implementation (Parallel CPU)
# -------------------------------------------------------------------------
def dask_complement(data):
    """
    Computes the DNA complement using Dask for out-of-core and parallel computation.
    """
    # Convert numpy array to Dask array with chunking
    chunk_size = 10_000_000  # 10M elements per chunk
    dask_data = da.from_array(data, chunks=chunk_size)
    
    # Define computation and execute (.compute() triggers actual evaluation)
    res = (3 - dask_data).compute()
    return res

# -------------------------------------------------------------------------
# Benchmarking Framework
# -------------------------------------------------------------------------
def run_benchmark(dataset_path):
    print(f"\n{'='*50}")
    print(f"Benchmarking Dataset: {dataset_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run 'python generate_datasets.py' first.")
        return

    # Load data
    start = time.time()
    data = np.load(dataset_path)
    load_time = time.time() - start
    print(f"Data loading time: {load_time:.4f} seconds (Size: {data.size:,} bases)")
    print(f"Data type: {data.dtype}, Memory footprint: {data.nbytes / (1024*1024):.2f} MB")
    print("-" * 50)

    # 1. CPU (NumPy) Baseline
    start = time.time()
    res_cpu = cpu_complement(data)
    cpu_time = time.time() - start
    print(f"[1] CPU (NumPy) Time:   {cpu_time:.4f} seconds")

    # 2. Numba CPU
    # Warm-up run (to exclude JIT compilation time from benchmark)
    _ = numba_cpu_complement(data[:1000])
    start = time.time()
    res_numba_cpu = numba_cpu_complement(data)
    numba_cpu_time = time.time() - start
    print(f"[2] Numba CPU Time:     {numba_cpu_time:.4f} seconds (Speedup vs CPU: {cpu_time/numba_cpu_time if numba_cpu_time>0 else 0:.2f}x)")

    # 3. Dask
    start = time.time()
    res_dask = dask_complement(data)
    dask_time = time.time() - start
    print(f"[3] Dask Time:          {dask_time:.4f} seconds (Speedup vs CPU: {cpu_time/dask_time if dask_time>0 else 0:.2f}x)")

    # 4. Numba GPU
    try:
        # Check if GPU is available
        if not cuda.is_available():
            print("[4] Numba GPU Time:     Skipped (CUDA GPU not available or not configured)")
        else:
            # Warm-up run (compile kernel and initialize context)
            _ = numba_gpu_complement(data[:1000])
            start = time.time()
            res_gpu = numba_gpu_complement(data)
            gpu_time = time.time() - start
            print(f"[4] Numba GPU Time:     {gpu_time:.4f} seconds (Speedup vs CPU: {cpu_time/gpu_time if gpu_time>0 else 0:.2f}x)")
            
            # Verify correctness
            if not np.array_equal(res_cpu, res_gpu):
                print("    WARNING: GPU result does not match CPU result!")
                
    except Exception as e:
        print(f"[4] Numba GPU Time:     Failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genome Processing Performance Benchmark")
    parser.add_argument("--dataset", type=str, default="all", choices=["small", "medium", "large", "all"], help="Dataset to benchmark")
    args = parser.parse_args()

    datasets = {
        "small": "datasets/small_genome.npy",
        "medium": "datasets/medium_genome.npy",
        "large": "datasets/large_genome.npy"
    }

    if args.dataset == "all":
        for name, path in datasets.items():
            if os.path.exists(path):
                run_benchmark(path)
            else:
                print(f"\nWarning: Dataset '{name}' ({path}) not found. Skipping.")
    else:
        run_benchmark(datasets[args.dataset])
