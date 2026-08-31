"""
numba_kernels.py – Numba CUDA kernel examples and utilities.

Provides
--------
vector_add(a, b)        : GPU element-wise vector addition (returns ndarray).
matrix_scale(mat, k)    : GPU in-place scalar multiply.
reduce_sum(arr)         : Simple parallel reduction sum.
benchmark_vs_numpy(n)   : Time comparison: GPU vs NumPy for vector_add.

All public functions accept plain NumPy arrays and return NumPy arrays.
Internally they copy to/from device memory automatically.

Notes
-----
On Windows WDDM, you must call ``numba.cuda.select_device(0)`` before
first use if ``cuda.is_available()`` returns False (Numba 0.59+ handles
this automatically when calling cuda.to_device, but we do it explicitly
for safety).
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from numba import cuda, float32, int32
    import numba
    _NUMBA_OK = True
except ImportError as _e:
    _NUMBA_OK = False
    _IMPORT_ERR = _e


# ---------------------------------------------------------------------------
# Internal: ensure device is selected (WDDM workaround)
# ---------------------------------------------------------------------------

_DEVICE_SELECTED = False


def _ensure_device(device_id: int = 0) -> None:
    """Select CUDA device once per process (WDDM-safe)."""
    global _DEVICE_SELECTED
    if _DEVICE_SELECTED:
        return
    if not _NUMBA_OK:
        raise ImportError(f"Numba not available: {_IMPORT_ERR}")
    cuda.select_device(device_id)
    _DEVICE_SELECTED = True


# ---------------------------------------------------------------------------
# Kernel: element-wise vector add
# ---------------------------------------------------------------------------

if _NUMBA_OK:
    @cuda.jit
    def _kernel_vector_add(a, b, out):
        """CUDA kernel: out[i] = a[i] + b[i]."""
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = a[i] + b[i]


def vector_add(a: np.ndarray, b: np.ndarray, *, device_id: int = 0) -> np.ndarray:
    """
    GPU element-wise addition of two 1-D arrays.

    Parameters
    ----------
    a, b : np.ndarray  – 1-D float32 arrays of the same length.
    device_id : int    – CUDA device index (default 0).

    Returns
    -------
    np.ndarray (float32)
    """
    _ensure_device(device_id)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    n = a.shape[0]
    threads = 256
    blocks = math.ceil(n / threads)

    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_out = cuda.device_array(n, dtype=np.float32)

    _kernel_vector_add[blocks, threads](d_a, d_b, d_out)
    cuda.synchronize()
    return d_out.copy_to_host()


# ---------------------------------------------------------------------------
# Kernel: matrix scalar scale (in-place)
# ---------------------------------------------------------------------------

if _NUMBA_OK:
    @cuda.jit
    def _kernel_matrix_scale(mat, scalar):
        """CUDA kernel: mat[i, j] *= scalar."""
        i, j = cuda.grid(2)
        if i < mat.shape[0] and j < mat.shape[1]:
            mat[i, j] *= scalar


def matrix_scale(mat: np.ndarray, k: float, *, device_id: int = 0) -> np.ndarray:
    """
    GPU in-place scalar multiplication of a 2-D matrix.

    Parameters
    ----------
    mat : np.ndarray  – 2-D float32 array.
    k   : float       – Scalar multiplier.

    Returns
    -------
    np.ndarray (float32) – scaled copy.
    """
    _ensure_device(device_id)
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("matrix_scale requires a 2-D array.")

    rows, cols = mat.shape
    threads = (16, 16)
    blocks = (math.ceil(rows / threads[0]), math.ceil(cols / threads[1]))

    d_mat = cuda.to_device(mat.copy())
    _kernel_matrix_scale[blocks, threads](d_mat, np.float32(k))
    cuda.synchronize()
    return d_mat.copy_to_host()


# ---------------------------------------------------------------------------
# Reduction: parallel sum
# ---------------------------------------------------------------------------

if _NUMBA_OK:
    @cuda.reduce
    def _gpu_sum(a, b):
        return a + b


def reduce_sum(arr: np.ndarray, *, device_id: int = 0) -> float:
    """
    GPU parallel reduction sum of a 1-D float32 array.

    Parameters
    ----------
    arr : np.ndarray – 1-D float32 array.

    Returns
    -------
    float – sum of all elements.
    """
    _ensure_device(device_id)
    arr = np.asarray(arr, dtype=np.float32)
    d_arr = cuda.to_device(arr)
    result = _gpu_sum(d_arr)
    return float(result)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

def benchmark_vs_numpy(
    n: int = 10_000_000,
    device_id: int = 0,
    repeat: int = 3,
) -> dict[str, float]:
    """
    Compare GPU vector_add vs NumPy for *n* elements over *repeat* runs.

    Returns
    -------
    dict with keys 'gpu_ms', 'numpy_ms', 'speedup'.
    """
    _ensure_device(device_id)

    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    # Warm up GPU
    _ = vector_add(a[:1024], b[:1024])

    gpu_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = vector_add(a, b)
        gpu_times.append(time.perf_counter() - t0)

    numpy_times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = a + b
        numpy_times.append(time.perf_counter() - t0)

    gpu_ms = min(gpu_times) * 1000
    numpy_ms = min(numpy_times) * 1000
    speedup = numpy_ms / gpu_ms if gpu_ms > 0 else float("inf")

    return {
        "n": n,
        "gpu_ms": round(gpu_ms, 3),
        "numpy_ms": round(numpy_ms, 3),
        "speedup": round(speedup, 2),
    }

# ---------------------------------------------------------------------------
# Heavy workload kernel (Dummy workload for scaling benchmark)
# ---------------------------------------------------------------------------

if _NUMBA_OK:
    @cuda.jit
    def _kernel_heavy_compute(arr, iterations):
        """CUDA kernel: repeatedly multiply and add to simulate heavy load."""
        i = cuda.grid(1)
        if i < arr.shape[0]:
            val = arr[i]
            for _ in range(iterations):
                val = (val * 1.01) + 0.01
                if val > 1000.0:
                    val = val / 1000.0
            arr[i] = val

def run_heavy_workload(chunk_array: np.ndarray, iterations: int = 10000, *, device_id: int = 0) -> np.ndarray:
    """
    Runs a heavy dummy workload on the GPU for the given sequence array.

    Parameters
    ----------
    chunk_array : np.ndarray – 1-D float32 array.
    iterations  : int – Number of dummy iterations per element.
    device_id   : int – CUDA device index (default 0).

    Returns
    -------
    np.ndarray (float32)
    """
    arr = np.asarray(chunk_array, dtype=np.float32)
    
    try:
        _ensure_device(device_id)
        n = arr.shape[0]
        threads = 256
        blocks = math.ceil(n / threads)

        d_arr = cuda.to_device(arr)
        _kernel_heavy_compute[blocks, threads](d_arr, iterations)
        cuda.synchronize()
        return d_arr.copy_to_host()
    except Exception as e:
        # Fallback to CPU if NVVM or CUDA is not fully installed
        import numba
        
        @numba.njit(nogil=True)
        def _cpu_heavy_compute(arr_in, iters):
            for i in range(arr_in.shape[0]):
                val = arr_in[i]
                for _ in range(iters):
                    val = (val * 1.01) + 0.01
                    if val > 1000.0:
                        val = val / 1000.0
                arr_in[i] = val
        
        _cpu_heavy_compute(arr, iterations)
        return arr
