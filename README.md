# GeneWeaver

GeneWeaver is a Python-based pipeline and toolkit for genomic data processing and GPU-accelerated computational tasks.

## Commit History & Repository Evolution

### 1. Initial Setup
- **Details**: Basic repository initialization and creation of the `README.md`.

### 2. Dask GPU Cluster Support
- **Details**: Introduced the `geneweaver` package with Dask GPU clustering capabilities.
  - Added `geneweaver/__init__.py` to expose the modules.
  - Added `geneweaver/dask_gpu.py` with helpers for creating Dask LocalClusters for both CPU and GPU execution (`make_local_cluster`, `make_gpu_cluster`). 
  - Includes gracefully handling of CPU fallbacks on Windows when GPU resources are constrained.
  - Provided demo scripts for processing Dask arrays and DataFrames.

### 3. GPU Environment Diagnostics
- **Details**: Added diagnostic tools for checking NVIDIA GPU health and CUDA runtime status.
  - Created `geneweaver/gpu_info.py` with helpers `probe()`, `print_summary()`, and `require_gpu()`.
  - Added WDDM-safe device selection workarounds for Windows to correctly probe NVIDIA devices using Numba.

### 4. Numba CUDA GPU-accelerated Kernels
- **Details**: Introduced numerical computing routines optimized for the GPU.
  - Added `geneweaver/numba_kernels.py` providing `vector_add`, `matrix_scale`, and `reduce_sum` via custom CUDA kernels.
  - Implemented `benchmark_vs_numpy` to compare computation speeds between GPU processing and standard NumPy routines.

### 5. GPU Environment Validation Script
- **Details**: Added utility script `scripts/check_env.py` to run sanity checks.
  - Automatically validates CUDA setup, Numba kernel compilation and execution, and Dask LocalCluster spin-up in a single pass.

### 6. BioPython FASTA Sequence Parsing
- **Details**: Implemented basic processing of mock genome data.
  - Added `mock_genome.fasta` as a test dataset.
  - Added `parse_fasta.py` to parse sequences and chunk them using BioPython.

### 7. Textual UI for Data Parsing
- **Details**: Upgraded the `parse_fasta.py` script with a Terminal User Interface (TUI).
  - Uses the `Textual` framework to render real-time progress bars and rich log output as the genome sequences are processed in chunks.

### 8. CLI Data Pipeline and JSON Output (Today's Commit)
- **Details**: Added `data_pipeline.py` script for automated processing without a UI.
  - Splits mock human genome FASTA files into chunked arrays.
  - Exports the chunked sequence data directly into JSON format (`chunked_genome.json`) for downstream processing pipelines.