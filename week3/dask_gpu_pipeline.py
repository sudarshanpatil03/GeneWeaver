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

