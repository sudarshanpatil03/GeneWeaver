#!/usr/bin/env python
"""
check_env.py – Quick sanity-check script for the geneweaver GPU environment.

Run with:
    python scripts/check_env.py          # full check
    python scripts/check_env.py --quick  # skip GPU kernel smoke test
"""

from __future__ import annotations

import sys
import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="geneweaver GPU environment checker")
    parser.add_argument("--quick", action="store_true",
                        help="Skip kernel smoke tests (faster, no GPU compute)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show verbose device attributes")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich import box
        console = Console()
    except ImportError:
        console = None

    def header(msg: str):
        if console:
            console.rule(f"[bold cyan]{msg}[/bold cyan]")
        else:
            print(f"\n{'='*60}\n  {msg}\n{'='*60}")

    def ok(msg: str):
        if console:
            console.print(f"  [green]✓[/green] {msg}")
        else:
            print(f"  [OK] {msg}")

    def warn(msg: str):
        if console:
            console.print(f"  [yellow]⚠[/yellow]  {msg}")
        else:
            print(f"  [WARN] {msg}")

    def fail(msg: str):
        if console:
            console.print(f"  [red]✗[/red] {msg}")
        else:
            print(f"  [FAIL] {msg}")

    # ── 1. GPU Info ──────────────────────────────────────────────
    header("1 · GPU / CUDA Environment")
    from geneweaver.gpu_info import probe, print_summary
    print_summary(verbose=args.verbose)
    info = probe()

    if info["gpu_usable"]:
        ok(f"GPU usable: {info['device_count']} device(s)")
    else:
        warn("No usable GPU detected (CPU-only mode).")

    # ── 2. Numba CUDA kernels ────────────────────────────────────
    header("2 · Numba CUDA Kernels")
    if not args.quick and info["gpu_usable"]:
        from geneweaver.numba_kernels import vector_add, reduce_sum, benchmark_vs_numpy
        import numpy as np

        # Small correctness check
        a = np.ones(1024, dtype=np.float32)
        b = np.full(1024, 2.0, dtype=np.float32)
        result = vector_add(a, b)
        if abs(result.mean() - 3.0) < 0.001:
            ok("vector_add kernel – correctness OK")
        else:
            fail(f"vector_add kernel – WRONG RESULT (mean={result.mean()})")
            return 1

        s = reduce_sum(np.ones(1024, dtype=np.float32))
        if abs(s - 1024.0) < 1.0:
            ok(f"reduce_sum kernel – result {s:.1f} (expected 1024.0)")
        else:
            fail(f"reduce_sum kernel – WRONG RESULT ({s})")
            return 1

        # Benchmark
        header("2b · GPU vs NumPy Benchmark (n=1M)")
        bench = benchmark_vs_numpy(n=1_000_000, repeat=3)
        if console:
            console.print(
                f"  GPU: [bold]{bench['gpu_ms']:.1f} ms[/bold]  |  "
                f"NumPy: [bold]{bench['numpy_ms']:.1f} ms[/bold]  |  "
                f"Speedup: [magenta]{bench['speedup']}×[/magenta]"
            )
        else:
            print(f"  GPU={bench['gpu_ms']}ms  NumPy={bench['numpy_ms']}ms  "
                  f"Speedup={bench['speedup']}x")
        ok("Benchmark complete")
    elif args.quick:
        warn("Skipped kernel tests (--quick flag).")
    else:
        warn("Skipped kernel tests (no GPU).")

    # ── 3. Dask cluster ─────────────────────────────────────────
    header("3 · Dask Local Cluster")
    from geneweaver.dask_gpu import make_local_cluster, demo_dask_array, demo_dask_dataframe, cluster_info
    from distributed import Client

    cluster = make_local_cluster(n_workers=2, threads_per_worker=1, memory_limit="1GB")
    try:
        client = Client(cluster)
        try:
            ci = cluster_info(client)
            ok(f"Cluster up: {ci['n_workers']} workers, "
               f"{ci['total_threads']} threads, "
               f"{ci['total_memory_gb']} GB total")
            ok(f"Dashboard: {ci['dashboard_link']}")

            # Dask array smoke test
            t0 = time.perf_counter()
            arr = demo_dask_array(size=100_000, chunks=10_000)
            elapsed = (time.perf_counter() - t0) * 1000
            ok(f"Dask array demo: shape={arr.shape}  ({elapsed:.1f} ms)")

            # Dask DataFrame smoke test
            df = demo_dask_dataframe(n_rows=50_000, n_partitions=4)
            ok(f"Dask DataFrame demo: {len(df)} groups – {df['group'].tolist()}")
        finally:
            client.close()
    finally:
        cluster.close()

    # ── 4. Summary ───────────────────────────────────────────────
    header("Environment Check Complete")
    if console:
        console.print("[bold green]All checks passed![/bold green] "
                      "geneweaver GPU environment is ready.")
    else:
        print("All checks passed! geneweaver GPU environment is ready.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
