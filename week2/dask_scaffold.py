"""
Week 3 Integration Scaffolding: Dask + GPU
This file prepares the structure for distributing GPU tasks across multiple workers using Dask.
"""

from dask.distributed import Client, LocalCluster
import dask.array as da
import numpy as np
import time

def gpu_task_wrapper(partition):
    """
    This function acts as a wrapper that Dask will call on each chunk of data.
    Inside this function, we will invoke the Numba CUDA kernels (to be integrated in Week 3).
    """
    # -------------------------------------------------------------
    # PLACEHOLDER FOR WEEK 3 CUDA INTEGRATION
    # Example Future Logic:
    # d_data = cuda.to_device(partition)
    # my_kernel[blocks, threads](d_data)
    # result = d_data.copy_to_host()
    # return result
    # -------------------------------------------------------------
    
    # For scaffolding, we simulate GPU transfer and processing time
    time.sleep(0.1)
    
    # Simulate some operation (like the complement from Week 1)
    result = 3 - partition
    return result

def setup_dask_cluster():
    """
    Initializes a local Dask cluster.
    For GPU workloads, we typically want 1 worker per GPU.
    """
    print("Initializing Dask LocalCluster...")
    # Scaffolding for local testing. 
    # In Week 3, this can be swapped with dask_cuda.LocalCUDACluster for multi-GPU setups.
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    print(f"Dashboard available at: {client.dashboard_link}")
    return client

def run_distributed_pipeline():
    """
    Example of how the Dask pipeline will be wired up.
    """
    client = setup_dask_cluster()
    
    try:
        print("\nCreating Dask Array (Mock Data)...")
        # Create a 1GB mock dataset chunked into 100MB pieces
        # Using dask array avoids loading the full 1GB into RAM instantly
        data = da.random.randint(0, 4, size=(1_000_000_000,), chunks=(100_000_000,), dtype=np.int8)
        print(f"Dataset Size: {data.nbytes / (1024**2):.2f} MB, Chunks: {data.npartitions}")
        
        print("Mapping GPU tasks across partitions...")
        # map_blocks applies our gpu_task_wrapper to every chunk in parallel
        result = data.map_blocks(gpu_task_wrapper, dtype=np.int8)
        
        print("Executing computation (triggering processing)...")
        start = time.time()
        # .compute() forces evaluation
        computed_result = result.compute()
        elapsed = time.time() - start
        
        print(f"Computation completed in {elapsed:.2f} seconds.")
        print(f"Result snippet: {computed_result[:10]}")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        print("Shutting down Dask cluster...")
        client.close()

if __name__ == "__main__":
    print("--- Dask + GPU Integration Scaffolding ---")
    run_distributed_pipeline()
