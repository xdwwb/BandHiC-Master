import time
import gc
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array, coo_array

import bandhic as bh
import hicstraw

data_path = '../data/GSE130275_mESC_WT_combined_1.3B_microc.hic'
# Resolutions to benchmark
RESOLUTIONS = [50000, 25000, 10000, 5000, 1000, 500]

MAX_DISTANCE = 2_000_000

# Number of random read/write queries per test (adjust if needed)
N_QUERIES = 10_000

RNG = np.random.default_rng(42)


def sizeof_sparse_coo(mat: coo_array) -> int:
    return mat.data.nbytes + mat.row.nbytes + mat.col.nbytes


def sizeof_sparse_csr(mat: csr_array) -> int:
    return mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes


def time_random_read(mat, row_idx, col_idx) -> float:
    t0 = time.perf_counter()
    for i in range(10000): # repeat fixed number for more stable timing
        _ = mat[row_idx, col_idx]
    return (time.perf_counter() - t0)/10000


def time_random_write(mat, row_idx, col_idx, value=1) -> float:
    t0 = time.perf_counter()
    for i in range(10000):
        mat[row_idx, col_idx] = value
    return (time.perf_counter() - t0)/10000

def benchmark_resolution(resolution: int) -> List[Dict]:
    print(f"\n=== Benchmarking resolution {resolution} ===", file=sys.stderr)
    mat_bh = bh.straw_chr(data_path,"1",resolution=resolution,diag_num=MAX_DISTANCE//resolution)
    n=mat_bh.shape[0]
    row_q = RNG.integers(0, n, size=N_QUERIES, dtype=np.int64)
    k_q = RNG.integers(0,MAX_DISTANCE//resolution, size=N_QUERIES,dtype=np.int64)
    col_q=row_q+k_q
    valid=col_q<n
    row_q=row_q[valid]
    col_q=col_q[valid]
    query_num=row_q.shape[0]

    results = []

    print('BandHiC')
    # BandHiC
    mem_bh = mat_bh.memory_usage()
    t_read = time_random_read(mat_bh, row_q, col_q)
    t_write = time_random_write(mat_bh, row_q, col_q, 1)
    results.append(dict(
        resolution=resolution,
        structure="BandHiC",
        memory_MiB=mem_bh / 1024**2,
        read_time_s=t_read,
        write_time_s=t_write,
        read_throughput=query_num/t_read,
        write_throughput=query_num/t_write
    ))
    
    print('COO')
    mat_coo = mat_bh.tocoo()
    del mat_bh
    gc.collect()
    # COO
    mem_coo = sizeof_sparse_coo(mat_coo)
    t_read = np.nan
    # COO write is inefficient; convert to CSR-like behavior via assignment
    t_write = np.nan
    results.append(dict(
        resolution=resolution,
        structure="COO",
        memory_MiB=mem_coo / 1024**2,
        read_time_s=t_read,
        write_time_s=t_write,
        read_throughput=query_num/t_read,
        write_throughput=query_num/t_write
    ))
    mat_csr=mat_coo.tocsr()
    del mat_coo
    gc.collect()

    print("CSR")
    # CSR
    mem_csr = sizeof_sparse_csr(mat_csr)
    t_read = time_random_read(mat_csr, row_q, col_q)
    t_write = time_random_write(mat_csr, row_q, col_q, 1)
    results.append(dict(
        resolution=resolution,
        structure="CSR",
        memory_MiB=mem_csr / 1024**2,
        read_time_s=t_read,
        write_time_s=t_write,
        read_throughput=query_num/t_read,
        write_throughput=query_num/t_write
    ))

    print('Dense')
    # Dense (NumPy matrix)
    # NOTE: This may fail for large n due to memory constraints
    if resolution <= 1000:
        results.append(dict(
            resolution=resolution,
            structure="Dense",
            memory_MiB=np.nan,
            read_time_s=np.nan,
            write_time_s=np.nan,
            read_throughput=np.nan,
            write_throughput=np.nan
        ))
        return results
    try:
        dense = mat_csr.todense()
        del mat_csr

        mem_dense = dense.nbytes
        t_read = time_random_read(dense, row_q, col_q)
        t_write = time_random_write(dense, row_q, col_q, 1.0)

        results.append(dict(
            resolution=resolution,
            structure="Dense",
            memory_MiB=mem_dense / 1024**2,
            read_time_s=t_read,
            write_time_s=t_write,
            read_throughput=query_num/t_read,
            write_throughput=query_num/t_write
        ))

        del dense
    except MemoryError:
        results.append(dict(
            resolution=resolution,
            structure="Dense",
            memory_MiB=np.nan,
            read_time_s=np.nan,
            write_time_s=np.nan,
            read_throughput=np.nan,
            write_throughput=np.nan
        ))
        
    return results


def main():
    all_results: List[Dict] = []
    for res in RESOLUTIONS:
        all_results.extend(benchmark_resolution(res))

    df = pd.DataFrame(all_results)
    out_path = "benchmark_sparse_results_mESC_throughput.csv"
    df.to_csv(out_path, index=False)
    print("\nBenchmark results:")
    print(df)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
