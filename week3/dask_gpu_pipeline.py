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

