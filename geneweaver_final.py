import os
import time
import numpy as np
from Bio import SeqIO
import dask
from dask.distributed import Client, LocalCluster, as_completed

try:
    from dask_cuda import LocalCUDACluster
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False

import numba
from numba import cuda

# 2-Bit Nucleotide Encoding Map: A=00, C=01, G=10, T=11
NUC_BIT_MAP = np.full(256, 0, dtype=np.uint8)
NUC_BIT_MAP[ord('A')] = 0b00
NUC_BIT_MAP[ord('C')] = 0b01
NUC_BIT_MAP[ord('G')] = 0b10
NUC_BIT_MAP[ord('T')] = 0b11

def pack_dna_sequence(dna_str: str) -> np.ndarray:
    ascii_bytes = np.frombuffer(dna_str.upper().encode('ascii'), dtype=np.uint8)
    mapped_2bit = NUC_BIT_MAP[ascii_bytes]
    pad_len = (4 - (len(mapped_2bit) % 4)) % 4
    if pad_len > 0:
        mapped_2bit = np.pad(mapped_2bit, (0, pad_len), 'constant', constant_values=0)
    packed = (mapped_2bit[0::4] << 6) | \
             (mapped_2bit[1::4] << 4) | \
             (mapped_2bit[2::4] << 2) | \
             (mapped_2bit[3::4])
    return packed.astype(np.uint8)

@cuda.jit
def crispr_offtarget_shared_kernel(
    d_genomic_packed,
    packed_len,
    genome_bp_len,
    target_packed,
    max_mismatches,
    d_out_indices,
    d_out_mismatches,
    d_out_scores,
    d_hit_count,
    global_offset
):
    # Shared memory optimization to load the 5-byte target once per block
    shared_target = cuda.shared.array(shape=(5,), dtype=numba.uint8)
    
    tid = cuda.threadIdx.x
    if tid < 5:
        shared_target[tid] = target_packed[tid]
    cuda.syncthreads()
    
    global_bp_idx = cuda.grid(1)
    if global_bp_idx + 23 > genome_bp_len:
        return
        
    byte_offset = global_bp_idx // 4
    bit_shift_offset = (global_bp_idx % 4) * 2
    if byte_offset + 6 >= packed_len:
        return
        
    raw_64 = numba.uint64(0)
    for b in range(7):
        raw_64 = (raw_64 << numba.uint64(8)) | numba.uint64(d_genomic_packed[byte_offset + b])
        
    shifted_window = raw_64 << numba.uint64(bit_shift_offset)
    sgrna_window = numba.uint64(shifted_window >> numba.uint64(24))
    
    target_64 = numba.uint64(0)
    for b in range(5):
        target_64 = (target_64 << numba.uint64(8)) | numba.uint64(shared_target[b])
        
    xor_diff = sgrna_window ^ target_64
    mismatches = 0
    score = 100
    
    for i in range(20):
        pair_diff = (xor_diff >> numba.uint64(38 - 2 * i)) & numba.uint64(0b11)
        if pair_diff != numba.uint64(0):
            mismatches += 1
            # Biological scoring logic: Mutations closer to PAM (index 19) are penalized more
            penalty = 5 if i > 15 else (2 if i > 10 else 1)
            score -= penalty
            if mismatches > max_mismatches:
                return
                
    pam_bp1 = (shifted_window >> numba.uint64(22)) & numba.uint64(0b11)
    pam_bp2 = (shifted_window >> numba.uint64(20)) & numba.uint64(0b11)
    if pam_bp1 == numba.uint64(0b10) and pam_bp2 == numba.uint64(0b10):
        hit_slot = cuda.atomic.add(d_hit_count, 0, 1)
        if hit_slot < d_out_indices.size:
            d_out_indices[hit_slot] = global_bp_idx + global_offset
            d_out_mismatches[hit_slot] = mismatches
            d_out_scores[hit_slot] = score if score > 0 else 0

def process_chunk_gpu(chunk_str: str, target_sgrna: str, max_mismatches: int, global_offset: int):
    if not cuda.is_available():
        raise RuntimeError("CUDA is not available on this worker.")
        
    packed_genome = pack_dna_sequence(chunk_str)
    packed_len = len(packed_genome)
    genome_bp_len = len(chunk_str)
    packed_target = pack_dna_sequence(target_sgrna)
    
    MAX_HITS = 10_000
    d_genomic = cuda.to_device(packed_genome)
    d_target = cuda.to_device(packed_target)
    d_out_indices = cuda.device_array(MAX_HITS, dtype=np.int32)
    d_out_mismatches = cuda.device_array(MAX_HITS, dtype=np.int32)
    d_out_scores = cuda.device_array(MAX_HITS, dtype=np.int32)
    d_hit_count = cuda.to_device(np.zeros(1, dtype=np.int32))
    
    THREADS_PER_BLOCK = 256
    blocks_per_grid = (genome_bp_len + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    
    crispr_offtarget_shared_kernel[blocks_per_grid, THREADS_PER_BLOCK](
        d_genomic,
        packed_len,
        genome_bp_len,
        d_target,
        max_mismatches,
        d_out_indices,
        d_out_mismatches,
        d_out_scores,
        d_hit_count,
        global_offset
    )
    cuda.synchronize()
    
    total_hits = d_hit_count.copy_to_host()[0]
    indices = d_out_indices[:total_hits].copy_to_host()
    mismatches = d_out_mismatches[:total_hits].copy_to_host()
    scores = d_out_scores[:total_hits].copy_to_host()
    
    results = []
    for i in range(total_hits):
        # Extract the sequence context from the chunk
        local_idx = indices[i] - global_offset
        context = chunk_str[local_idx:local_idx + 23] if local_idx + 23 <= len(chunk_str) else "N/A"
        results.append({
            'index': int(indices[i]),
            'mismatches': int(mismatches[i]),
            'score': int(scores[i]),
            'context': context
        })
    return results

def process_chunk_cpu(chunk_str: str, target_sgrna: str, max_mismatches: int, global_offset: int):
    """CPU fallback when no GPU is available."""
    query = target_sgrna + "AGG"
    m = len(query)
    results = []
    for i in range(len(chunk_str) - m + 1):
        mismatches = 0
        score = 100
        for j in range(20):
            if chunk_str[i + j] != query[j]:
                mismatches += 1
                penalty = 5 if j > 15 else (2 if j > 10 else 1)
                score -= penalty
                if mismatches > max_mismatches:
                    break
        if mismatches <= max_mismatches and chunk_str[i + 21:i + 23] == "GG":
            context = chunk_str[i:i + 23]
            results.append({
                'index': i + global_offset,
                'mismatches': mismatches,
                'score': max(0, score),
                'context': context
            })
    return results

def setup_dask_cluster():
    import sys
    if HAS_DASK_CUDA:
        cluster = LocalCUDACluster()
    else:
        # On Windows, processes=False avoids the multiprocessing spawn crash
        use_processes = sys.platform != 'win32'
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=use_processes)
    return Client(cluster), cluster

def parse_fasta_chunks(fasta_path: str, chunk_size_bp: int = 20_000_000):
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence = str(record.seq).upper()
        # Handle overlap of 22 bp to not miss targets spanning chunks
        overlap = 22
        for i in range(0, len(sequence), chunk_size_bp):
            end = min(i + chunk_size_bp + overlap, len(sequence))
            chunk = sequence[i:end]
            yield chunk, i

def run_pipeline(fasta_path: str, target_sgrna: str, max_mismatches: int = 4):
    client, cluster = setup_dask_cluster()
    
    try:
        chunks = list(parse_fasta_chunks(fasta_path))
        total_chunks = len(chunks)
        
        # Select GPU or CPU processing function based on hardware
        has_gpu = cuda.is_available()
        process_fn = process_chunk_gpu if has_gpu else process_chunk_cpu
        
        yield {"status": "init", "total_chunks": total_chunks, "dashboard": client.dashboard_link, "gpu": has_gpu}
        
        futures = []
        for chunk, offset in chunks:
            f = client.submit(process_fn, chunk, target_sgrna, max_mismatches, offset)
            futures.append(f)
            
        completed = 0
        all_results = []
        
        for future in as_completed(futures):
            res = future.result()
            all_results.extend(res)
            completed += 1
            yield {"status": "progress", "completed": completed, "total": total_chunks, "results": res}
            
        yield {"status": "done", "final_results": all_results}
        
    except Exception as e:
        yield {"status": "error", "message": str(e)}
    finally:
        client.close()
        cluster.close()
