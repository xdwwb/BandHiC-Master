import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hicstraw
import tracemalloc
import bandhic as bh
from bandhic import call_tad
import multiprocessing
import time

def straw_chr_dense(
    hic_file: str,
    chrom: str,
    resolution: int,
    diag_num: int,
    data_type: str = "observed",
    normalization: str = "NONE",
    unit: str = "BP",
) -> np.ndarray:
    """
    Read Hi-C data from a .hic file and return a band_hic_matrix.

    Parameters
    ----------
    hic_file : str
        Path to the .hic file. This file should be in the Hi-C format compatible with hicstraw. Local or remote paths are supported.
    chrom : str
        Chromosome name (e.g., 'chr1', 'chrX'). Short names like '1', 'X' are also accepted.
    resolution : int
        Resolution of the Hi-C data. Such as 10000 for 10kb resolution.
    diag_num : int
        Number of diagonals to consider.
    data_type : str, optional
        Type of data to read from the Hi-C file. Default is 'observed'. Other options include 'expected', 'balanced', etc.
        See `hicstra`w` documentation for more details.
    normalization : str, optional
        Normalization method to apply. Default is 'NONE'. Other options include 'VC', 'VC_SQRT', 'KR', 'SCALE', etc.
        See `hicstraw` documentation for more details.
    unit : str, optional
        Unit of measurement for the Hi-C data. Default is 'BP' (base pairs). Other options include 'FRAG' (fragments), etc.
        
    See also
    --------
    `hicstraw` documentation for more details on available parameters and usage.
    URL: https://github.com/aidenlab/straw/tree/master/pybind11_python

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object containing the Hi-C data.

    Raises
    ------
    ValueError
        If the file cannot be parsed or parameters are invalid.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = straw_chr_dense('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', 'chr1', resolution=10000, diag_num=200)
    >>> isinstance(mat, np.ndarray)
    True
    """
    chrom_short = chrom.replace("chr", "") if chrom.startswith("chr") else chrom
    records = hicstraw.straw(
        data_type,
        normalization,
        hic_file,
        chrom_short,
        chrom_short,
        unit,
        resolution,
    )

    row_idx = np.array(
        [record.binX // resolution for record in records]
    )
    col_idx = np.array(
        [record.binY // resolution for record in records]
    )
    coo_data = np.array([record.counts for record in records])
    
    n_bins = max(row_idx.max(), col_idx.max())+1
    
    dense_data = np.zeros((n_bins, n_bins))
    
    dense_data[row_idx, col_idx] = coo_data

    return dense_data

def detect_TADs_bandhic(resolution, chrom, queue):
    tic = time.time()
    mat = bh.straw_chr('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', 'chr'+chrom, resolution=resolution, diag_num=2000000//resolution)
    tic2 = time.time()
    memory_usage = mat.memory_usage()
    tracemalloc.start()
    # domains, bins = detect_TADs(mat, resolution, chrom, 250000, 100000)
    domains, bins = call_tad(mat, resolution, chrom, 250000, 100000)
    current, peak = tracemalloc.get_traced_memory()
    print("memory usage:",current/1024/1024,"MB")
    print("peak memory usage:",peak/1024/1024,"MB")
    tracemalloc.stop()
    tic3 = time.time()
    toc = time.time()
    print("band_hic_matrix","chrom",chrom,":",domains.shape[0],"whole time:",toc-tic," straw time:",tic2-tic,"topdom time:",tic3-tic2,"memory usage:",memory_usage/1024/1024,"MB","peak memory usage:",peak/1024/1024,"MB")
    result = ['band_hic_matrix',chrom,domains.shape[0],mat.shape[0],toc-tic,tic2-tic,tic3-tic2,memory_usage,peak]
    print("--------------------------------")
    queue.put(result)

def detect_TADs_dense(resolution, chrom, queue):
    tic = time.time()
    mat = straw_chr_dense('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', 'chr'+chrom, resolution=resolution, diag_num=2000)
    tic2 = time.time()
    memory_usage = mat.nbytes
    tracemalloc.start()
    # domains, bins = detect_TADs(mat, resolution, chrom, 250000, 100000)
    domains, bins = call_tad(mat, resolution, chrom, 250000, 100000)
    current, peak = tracemalloc.get_traced_memory()
    print("memory usage:",current/1024/1024,"MB")
    print("peak memory usage:",peak/1024/1024,"MB")
    tracemalloc.stop()
    tic3 = time.time()
    toc = time.time()
    print("dense_matrix","chrom",chrom,":",domains.shape[0],"whole time:",toc-tic," straw time:",tic2-tic,"topdom time:",tic3-tic2,"memory usage:",memory_usage/1024/1024,"MB","peak memory usage:",peak/1024/1024,"MB")
    result = ['dense_matrix',chrom,domains.shape[0],mat.shape[0],toc-tic,tic2-tic,tic3-tic2,memory_usage,peak]
    print("--------------------------------")
    queue.put(result)

if __name__ == "__main__":
    results = []
    for resolution in [50000]:
        # for chrom in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','X']:
        for chrom in ['19']:
            # resolution = 25000
            queue1 = multiprocessing.Queue()
            queue2 = multiprocessing.Queue()
            p1 = multiprocessing.Process(target=detect_TADs_bandhic, args=(resolution, chrom, queue1))
            p1.start()
            p1.join()
            if not queue1.empty():
                results.append(queue1.get())
            else:
                results.append(['band_hic_matrix',chrom,0,0,0,0,0,0,0])
            p2 = multiprocessing.Process(target=detect_TADs_dense, args=(resolution, chrom, queue2))
            p2.start()
            p2.join()
            if not queue2.empty():
                results.append(queue2.get())
            else:
                results.append(['dense_matrix',chrom,0,0,0,0,0,0,0])
 
    results = pd.DataFrame(results,columns=['matrix_type','chrom','n_domains','n_bins','whole_time','straw_time','topdom_time','matrix_memory_usage','peak_memory_usage'])
    results.to_csv('topdom_results_test.csv',index=False)