# GeneWeaver: GPU-Accelerated CRISPR Off-Target Alignment Engine

**Domain:** Bioinformatics & High-Performance Computing (HPC)

---

## What Does GeneWeaver Do?

GeneWeaver scans massive genomic FASTA files to find **unintended CRISPR off-target mutations**. It takes a 20-base-pair sgRNA (single guide RNA) sequence and searches the entire genome for locations where that sequence (plus the NGG PAM site) appears with up to N mismatches. These are potential sites where CRISPR could accidentally cut.

Instead of running this search on a CPU (which can take hours), GeneWeaver JIT-compiles a custom CUDA kernel via Numba and runs millions of comparisons simultaneously on your GPU. A Dask scheduler distributes the work across all available GPUs.

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     main.py (Textual TUI)                │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ Progress   │  │  Log Panel   │  │  Results Table   │ │
│  │ Bar        │  │  (live logs) │  │  (mismatches in  │ │
│  │            │  │              │  │   red highlight) │ │
│  └────────────┘  └──────────────┘  └──────────────────┘ │
└───────────────────────┬──────────────────────────────────┘
                        │ calls run_pipeline()
                        ▼
┌──────────────────────────────────────────────────────────┐
│              geneweaver_final.py (Backend Engine)         │
│                                                          │
│  1. BioPython Parser ──► Chunks FASTA into 20M bp pieces │
│  2. Dask Scheduler   ──► Distributes chunks to workers   │
│  3. CUDA Kernel      ──► Bitwise XOR alignment per GPU   │
│  4. Scoring Matrix   ──► Ranks mutation severity by PAM  │
└──────────────────────────────────────────────────────────┘
```

---

## How to Run

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA drivers installed (falls back to CPU if unavailable)
- Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Place Your FASTA File

Put your `.fasta` or `.fa` file in the same directory as `main.py`. You can use the included `mock_genome.fasta` for a quick test, or use any real genome FASTA file downloaded from public databases like [NCBI](https://www.ncbi.nlm.nih.gov/datasets/genome/), [UCSC](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/), or [Ensembl](http://ftp.ensembl.org/pub/current_fasta/).

### Step 2: Understand the Key Inputs

Before running, you need to understand two critical biological parameters:

#### What is `--sgrna` (the 20bp target)?
In CRISPR gene editing, a **sgRNA (single guide RNA)** is a 20-character DNA sequence made up of the letters A, C, G, and T. It acts as the "address" that tells the CRISPR-Cas9 protein exactly where to cut the genome. The researcher designs this sequence to match a specific gene they want to edit.

**Example:** `GAGTCCGAGCAGAAGAAGAA` — this is a 20 base-pair sequence that the CRISPR system will use to find its target location in the genome.

The problem is that this 20-letter sequence might also appear (with slight variations) at **other unintended locations** in the genome. These are called **off-target sites**, and they are dangerous because CRISPR could accidentally cut DNA at these wrong locations, causing harmful mutations.

**GeneWeaver scans the entire genome to find all these off-target sites.**

#### What is `--mismatches`?
A **mismatch** is a position where the genome sequence differs from the sgRNA target by one letter. For example:

```
Target sgRNA: GAGTCCGAGCAGAAGAAGAA
Genome site:  GCGTCCGACTAGAAGAAGAA
               ^      ^^           ← 3 mismatches (positions marked with ^)
```

CRISPR can still accidentally cut at a site even if it doesn't perfectly match the sgRNA — it tolerates a few mismatches. The `--mismatches` parameter controls how many mismatches are allowed:

| Value | Meaning                                          | Use Case                    |
|-------|--------------------------------------------------|-----------------------------|
| `0`   | Only find **exact matches**                      | Very strict, fewest results |
| `1-2` | Find sites with 1-2 letter differences           | Moderate search             |
| `3-4` | Find sites with up to 3-4 letter differences     | Comprehensive safety scan   |
| `4`   | **(Default)** Most thorough off-target detection | Recommended for research    |

### Step 3: Run the Application

```bash
python main.py --fasta <your_file.fasta> --sgrna <your_20_letter_target> --mismatches <0_to_4>
```

**Example with mock data:**
```bash
python main.py --fasta mock_genome.fasta --sgrna GAGTCCGAGCAGAAGAAGAA --mismatches 4
```

**Example with real chromosome data:**
```bash
python main.py --fasta chr1.fa --sgrna GAGTCCGAGCAGAAGAAGAA --mismatches 4
```

### Command-Line Arguments

| Argument       | Default                    | Description                                                                 |
|----------------|----------------------------|-----------------------------------------------------------------------------|
| `--fasta`      | `mock_genome.fasta`        | Path to the input FASTA file containing genome data                         |
| `--sgrna`      | `GAGTCCGAGCAGAAGAAGAA`     | The 20-letter sgRNA target sequence (only A, C, G, T characters)            |
| `--mismatches` | `4`                        | Max allowed letter differences between target and genome site (0-4)         |

---

## How the Pipeline Works (Step by Step)

### 1. FASTA Parsing (BioPython)
The `parse_fasta_chunks()` function uses BioPython's `SeqIO.parse()` to read the FASTA file record by record. Each chromosome's sequence is split into **20 million base-pair chunks** with a **22 bp overlap** between adjacent chunks. The overlap ensures that no potential 23 bp target site (20 bp sgRNA + 3 bp PAM) is missed at chunk boundaries.

### 2. Dask Distribution
The `run_pipeline()` function initializes a Dask cluster:
- If `dask_cuda` is installed → uses `LocalCUDACluster` (one worker per GPU, automatic GPU assignment).
- Otherwise → uses `LocalCluster` with one worker per detected GPU.

Each chunk is submitted to the cluster as an independent task via `client.submit()`. Dask automatically load-balances across all available workers/GPUs.

### 3. 2-Bit DNA Packing
Before GPU processing, each chunk is compressed using a **2-bit encoding scheme**:
- `A = 00`, `C = 01`, `G = 10`, `T = 11`
- 4 nucleotides are packed into a single byte → **4× memory compression**

This reduces VRAM consumption and increases memory bandwidth utilization on the GPU.

### 4. CUDA Kernel Execution (Shared Memory Optimized)
The `crispr_offtarget_shared_kernel` is JIT-compiled by Numba's `@cuda.jit` decorator. Each GPU thread evaluates one position in the genome:

1. **Shared Memory Loading**: The first 5 threads in each block cooperatively load the 5-byte packed target sequence into **CUDA shared memory**. This is ~100× faster than global VRAM access.
2. **Bitwise XOR Comparison**: The thread extracts a 23 bp window from the packed genome and XORs it against the target. Each non-zero 2-bit pair in the XOR result = one mismatch.
3. **Early Exit**: If mismatches exceed the threshold, the thread returns immediately (no wasted cycles).
4. **PAM Verification**: Only if mismatches ≤ threshold, the thread checks whether positions 21-22 encode `GG` (the NGG PAM pattern).
5. **Atomic Write**: Valid hits are written to the output buffer using `cuda.atomic.add()` to avoid race conditions.

### 5. Biological Scoring
Each hit receives a severity score (0-100) based on **PAM-proximal weighting**:
- Mismatches at positions 16-19 (closest to PAM) → **-5 points each** (most dangerous)
- Mismatches at positions 11-15 → **-2 points each**
- Mismatches at positions 0-10 → **-1 point each** (least dangerous)

This reflects the biological reality that CRISPR is more tolerant of mismatches at the PAM-distal end of the guide RNA.

### 6. TUI Dashboard (Textual)
The `main.py` Textual app renders a live terminal dashboard:
- **Left Panel**: Progress bar, Dask dashboard link, and streaming log output.
- **Right Panel**: DataTable showing each hit's genome position, the 23 bp context sequence (with **mutated bases highlighted in red**), mismatch count, and color-coded severity score.

---

## Frequently Asked Questions

### Q: Where do I place my FASTA file?
**A:** Place it in the same directory as `main.py` (the `Final_Code_Working` folder). Then pass the filename via the `--fasta` argument.

### Q: What FASTA format is supported?
**A:** Standard multi-record FASTA. Each record starts with a `>header` line followed by lines of nucleotide characters (A, C, G, T). Example:
```
>chr1 Homo sapiens chromosome 1
ATGCGTGCATGCATGCATGCATGCATGCATGCATGC...
>chr2 Homo sapiens chromosome 2
CGTACGTACGTACGTACGTACGTACGTACGTACGTA...
```

### Q: How do I get a real FASTA file to test?
**A:** Download a chromosome FASTA file from any of these free public databases:
- [UCSC Genome Browser](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/) — download `chr22.fa.gz` (~12 MB) or `chr1.fa.gz` (~70 MB, extracts to ~242 MB)
- [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/genome/) — search any organism, click Download → Genomic Sequence (FASTA)
- [Ensembl FTP](http://ftp.ensembl.org/pub/current_fasta/) — navigate to `homo_sapiens/dna/`

Place the extracted `.fa` or `.fasta` file in this folder and pass it via `--fasta`.

### Q: How does the genome get chunked?
**A:** The FASTA file is read by BioPython and each chromosome sequence is split into chunks of 20 million base pairs. Each chunk overlaps the next by 22 bp so that target sites spanning chunk boundaries are not missed.

### Q: What happens if I don't have a GPU?
**A:** The pipeline will detect that CUDA is unavailable and automatically fall back to a CPU-based Dask cluster. It will still work, just slower.

### Q: How does the Dask pipeline distribute work?
**A:** Dask creates one worker per GPU. Each genomic chunk is submitted as an independent `Future`. The Dask scheduler assigns chunks to idle workers. Results stream back as they complete via `as_completed()`, which updates the TUI in real time.

### Q: What is CUDA Shared Memory and why does it matter?
**A:** Every CUDA thread block has a small, fast on-chip memory (~48 KB) called shared memory. Instead of each of the 256 threads in a block independently fetching the target sequence from slow global VRAM, we load it once into shared memory. All 256 threads then read from shared memory at near-register speed. This reduces redundant VRAM reads by 256×.

### Q: How is the severity score calculated?
**A:** A perfect match scores 100. Each mismatch deducts points based on proximity to the PAM site:
| Position Range | Penalty | Biological Reason                |
|----------------|---------|----------------------------------|
| 0-10           | -1      | PAM-distal, CRISPR tolerates     |
| 11-15          | -2      | Moderate sensitivity             |
| 16-19          | -5      | PAM-proximal, highly sensitive   |

Scores below 50 are marked **red** (critical), 50-74 are **orange** (moderate), and 75+ are **green** (low risk).

### Q: What does the red highlighting in the results table mean?
**A:** Each base pair in the 23 bp context string is compared against the expected target+PAM sequence. Any base that differs (a mismatch) is rendered in **bold red** text in the terminal. This lets you instantly see which nucleotides mutated.

### Q: Can this handle the full human genome (3.2 billion bp)?
**A:** Yes. The chunking + Dask distribution architecture is designed for exactly this. The genome is never loaded entirely into RAM. Each 20M bp chunk is processed independently on the GPU and results are streamed back. Memory usage stays bounded regardless of genome size.

---

## File Structure

```
Final_Code_Working/
├── main.py                  # Textual TUI entry point
├── geneweaver_final.py      # Backend engine (CUDA + Dask + BioPython)
├── requirements.txt         # Python dependencies
├── mock_genome.fasta        # Small test FASTA file (372 bytes)
└── chr1.fa                  # Real Human Chromosome 1 FASTA data (~242 MB)
```

---

## Tech Stack

| Component              | Technology           | Purpose                                    |
|------------------------|----------------------|--------------------------------------------|
| Genomic Parser         | BioPython (SeqIO)    | Reads and chunks .fasta files              |
| CUDA Kernel Engine     | Numba (@cuda.jit)    | GPU-accelerated bitwise alignment          |
| Distributed Scheduler  | Dask Distributed     | Multi-GPU workload distribution            |
| TUI Dashboard          | Textual              | Live terminal interface with progress      |
| DNA Compression        | NumPy 2-bit packing  | 4× memory reduction for GPU transfer       |
