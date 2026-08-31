import numpy as np
import time
import argparse
import os
import dask.array as da
from dask.distributed import Client, LocalCluster, wait
import numba
from numba import cuda
import warnings

# Suppress some Numba warnings for cleaner output
warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)

try:
    from dask_cuda import LocalCUDACluster
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False
    print("Warning: dask_cuda not found. Falling back to LocalCluster. Multi-GPU balancing might not work perfectly.")

# -------------------------------------------------------------------------
# 1. Numba CUDA Kernel
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

def gpu_task_wrapper(partition):
    """
    Wrapper function that Dask will call on each chunk of data.
    """
    # If the partition is empty or not an ndarray, handle it gracefully
    if partition.size == 0:
        return partition

    # Allocate device memory and copy data from host (CPU) to device (GPU)
    # The worker process will automatically use its assigned GPU
    d_data = cuda.to_device(partition)
    d_res = cuda.device_array_like(partition)
    
    # Configure grid and block dimensions
    threadsperblock = 256
    blockspergrid = (partition.size + (threadsperblock - 1)) // threadsperblock
    
    # Launch kernel
    numba_gpu_complement_kernel[blockspergrid, threadsperblock](d_data, d_res)
    
    # Wait for the GPU to finish execution
    cuda.synchronize()
    
    # Copy the result back from device (GPU) to host (CPU)
    return d_res.copy_to_host()

# -------------------------------------------------------------------------
# 2. Dask Cluster Setup
# -------------------------------------------------------------------------
def setup_dask_cluster(num_gpus=None):
    """
    Initializes a local Dask cluster utilizing GPUs.
    """
    print("Initializing Dask Cluster...")
    
    if HAS_DASK_CUDA:
        # LocalCUDACluster creates one worker per GPU by default, setting CUDA_VISIBLE_DEVICES
        cluster = LocalCUDACluster()
        
        # If user explicitly wants to test with fewer GPUs (e.g., 1 GPU for scaling baseline)
        if num_gpus is not None:
            # We scale down the cluster to the requested number of workers (GPUs)
            cluster.scale(num_gpus)
            
        client = Client(cluster)
        active_workers = len(client.scheduler_info()['workers'])
        print(f"Initialized LocalCUDACluster with {active_workers} GPU workers.")
        
    else:
        # Fallback to a standard LocalCluster
        # If num_gpus is None, detect via Numba
        if num_gpus is None:
            try:
                num_gpus = len(cuda.gpus)
            except Exception:
                num_gpus = 1
                
        # Limit n_workers to num_gpus to mimic 1 worker per GPU
        cluster = LocalCluster(n_workers=max(1, num_gpus), threads_per_worker=1)
        client = Client(cluster)
        print(f"Initialized fallback LocalCluster with {max(1, num_gpus)} workers.")
        
    print(f"Dashboard available at: {client.dashboard_link}")
    return client, cluster

# -------------------------------------------------------------------------
# 3. Pipeline Execution
# -------------------------------------------------------------------------
def run_benchmark(data_size, num_gpus=None):
    client, cluster = setup_dask_cluster(num_gpus)
    
    try:
        # Ensure workers are ready
        client.wait_for_workers(n_workers=num_gpus if num_gpus is not None else 1)
        active_workers = len(client.scheduler_info()['workers'])
        print(f"Running benchmark with {active_workers} GPU(s)")
        
        # Determine chunk size. We want chunks to be distributed evenly across GPUs.
        # A good rule of thumb is a few chunks per worker to balance load but avoid scheduling overhead.
        # Let's say 100MB per chunk. np.int8 is 1 byte, so 100,000,000 elements.
        chunk_size = 100_000_000
        
        print(f"\nCreating Dask Array of size {data_size:,} bases...")
        # Simulate loading data
        # We use random.randint to generate sequence data (0 to 3)
        data = da.random.randint(0, 4, size=(data_size,), chunks=(chunk_size,), dtype=np.int8)
        
        # Map the GPU wrapper function across all partitions
        # We use map_blocks because we want our function to receive each NumPy chunk
        result = data.map_blocks(gpu_task_wrapper, dtype=np.int8)
        
        print(f"Dataset Size: {data.nbytes / (1024**2):.2f} MB, Chunks: {data.npartitions}")
        print("Executing multi-GPU computation...")
        
        start = time.time()
        # compute() triggers the execution across the Dask cluster
        computed_result = result.compute()
        elapsed = time.time() - start
        
        print(f"Computation completed in {elapsed:.4f} seconds.")
        print(f"Throughput: {data.nbytes / (1024**3) / elapsed:.4f} GB/s")
        
        # Very basic validation
        sample_in = data[:10].compute()
        sample_out = computed_result[:10]
        print(f"Validation: (Input + Output should equal 3)")
        print(f"Input : {sample_in}")
        print(f"Output: {sample_out}")
        
        return elapsed
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        return None
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask Multi-GPU Benchmark")
    parser.add_argument("--size", type=int, default=1_000_000_000, help="Number of bases to process (default 1 Billion)")
    args = parser.parse_args()

    print("--- Week 3: Dask + Multi-GPU Integration ---")
    
    # Check if CUDA is available at all
    try:
        if not cuda.is_available():
            print("Error: CUDA is not available. Please ensure GPUs are configured.")
            exit(1)
        total_gpus = len(cuda.gpus)
        print(f"Detected {total_gpus} CUDA device(s).")
    except Exception as e:
        print(f"Error detecting CUDA devices: {e}")
        exit(1)
        
    if total_gpus > 1:
        print("\n--- Running 1 GPU Baseline ---")
        time_1gpu = run_benchmark(args.size, num_gpus=1)
        
        print(f"\n--- Running All ({total_gpus}) GPUs ---")
        time_allgpus = run_benchmark(args.size, num_gpus=total_gpus)
        
        if time_1gpu and time_allgpus:
            speedup = time_1gpu / time_allgpus
            print(f"\n--- Scaling Results ---")
            print(f"1 GPU Time   : {time_1gpu:.4f} s")
            print(f"{total_gpus} GPUs Time: {time_allgpus:.4f} s")
            print(f"Speedup      : {speedup:.2f}x")
            
            ideal_speedup = total_gpus
            efficiency = speedup / ideal_speedup * 100
            print(f"Efficiency   : {efficiency:.2f}%")
    else:
        print("\n--- Running Single GPU Execution ---")
        run_benchmark(args.size, num_gpus=1)
