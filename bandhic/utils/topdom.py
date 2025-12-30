# -*- coding: utf-8 -*-
# topdom.py

"""
TopDom: A Python implementation of the TopDom algorithm for detecting TADs (Topologically Associating Domains) in Hi-C data.
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

This module provides functions to detect TADs from Hi-C matrices using the TopDom algorithm.
It is translated from the original R implementation and adapted for use with the `band_hic_matrix` class from the BandHiC package.

Reference:
1. Shin, H. et al. TopDom: an efficient and deterministic method for identifying topological domains in genomes. Nucleic Acids Research 44, e70 (2016). https://doi.org/10.1093/nar/gkw064

"""

import numpy as np
import pandas as pd
import time
import scipy.stats as stats
import scipy.sparse as sparse
import bandhic as bh
from typing import Union

__all__ = [
    "call_tad",
    "topdom",
]


def call_tad(
    hic_matrix: Union[bh.band_hic_matrix, np.ndarray],
    resolution: int,
    chrom_short: str,
    window_size_bp: int = 200_000,
    min_TAD_size: int = None,
    stat_filter: bool = True,
    verbose: bool = False,
):
    """
    Detect TADs from a Hi-C matrix using the TopDom algorithm. Different from the `TopDom` function, this function accepts a Hi-C matrix in either `band_hic_matrix` format or as a dense numpy array.
    This function is designed to be used with the BandHiC package and provides a convenient interface for TAD detection.

    Parameters
    ----------
    hic_matrix : band_hic_matrix or np.ndarray
        The Hi-C matrix, either as a band_hic_matrix object or a dense numpy array.
    resolution : int
        The resolution of the Hi-C matrix.
    chrom_short : str
        The chromosome name without 'chr' prefix.
    window_size_bp : int, optional
        The size of the window to consider for detecting TADs, in base pairs. Default is 200000.
    min_TAD_size : int, optional
        The minimum size of a TAD to be considered valid, in base pairs. Default is None.
    stat_filter : bool, optional
        Whether to apply statistical filtering to remove false positives. Default is True.
    verbose : bool, optional
        Whether to print detailed information during processing. Default is False.

    Returns
    -------
    domains : pd.DataFrame
        DataFrame containing detected TADs with columns 'chr', 'from.id', 'from.coord',
        'to.id', 'to.coord', 'tag'.
    bins : pd.DataFrame
        DataFrame containing bin information columns 'id', 'chr', 'from.coord', 'to.coord', 'local.ext', 'mean.cf', 'pvalue'.
    """
    n = hic_matrix.shape[0]
    from_cord = np.arange(0, n) * resolution
    to_cord = np.arange(1, n + 1) * resolution
    chrom = np.repeat("chr" + chrom_short, n)
    bins = pd.DataFrame(
        {
            "id": np.arange(n),
            "chr": chrom,
            "from.coord": from_cord,
            "to.coord": to_cord,
        }
    )

    domains, bins = topdom(
        hic_matrix,
        bins,
        window_size_bp // resolution,
        verbose=False,
        stat_filter=stat_filter,
    )
    if min_TAD_size is None:
        return domains, bins
    if not isinstance(min_TAD_size, int):
        raise ValueError(
            "min_TAD_size must be an integer representing the minimum TAD size in base pairs."
        )
    if min_TAD_size < resolution:
        raise ValueError(
            "min_TAD_size must be greater than or equal to the resolution of the Hi-C matrix."
        )
    # Merge small TADs
    merged_TAD = domains.copy()
    left_index = []
    merged_TAD["size_bin"] = merged_TAD["to.id"] - merged_TAD["from.id"]
    merged_TAD.reset_index(drop=True, inplace=True)
    for index in merged_TAD.index:
        row = merged_TAD.loc[index, :]
        if row.size_bin < min_TAD_size // resolution and row.tag != "gap":
            if left_index:
                last_index = left_index[-1]
                last_tag = merged_TAD.at[last_index, "tag"]
                last_mean_cf = bins.at[row["from.id"], "mean.cf"]
            else:
                last_index = None
                last_tag = None

            if index + 1 < merged_TAD.shape[0]:
                next_tag = merged_TAD.at[index + 1, "tag"]
                next_mean_cf = bins.at[row["to.id"], "mean.cf"]
            else:
                next_tag = None
            if last_tag == "gap" and next_tag == "gap":
                left_index.append(index)
                continue
            elif last_tag == "gap" or not last_index:
                # Merge to next TAD
                merged_TAD.at[index + 1, "from.id"] = row["from.id"]
                merged_TAD.at[index + 1, "from.coord"] = row["from.coord"]
                merged_TAD.at[index + 1, "size_bin"] += row["size_bin"]
            elif next_tag == "gap":
                # Merge to last TAD
                if last_index:
                    merged_TAD.at[last_index, "to.id"] = row["to.id"]
                    merged_TAD.at[last_index, "to.coord"] = row["to.coord"]
                    merged_TAD.at[last_index, "size_bin"] += row["size_bin"]
                else:
                    left_index.append(index)
                    continue
            else:
                # Merge based on mean contact frequency
                if last_mean_cf >= next_mean_cf:
                    # Merge to last TAD
                    if last_index:
                        merged_TAD.at[last_index, "to.id"] = row["to.id"]
                        merged_TAD.at[last_index, "to.coord"] = row["to.coord"]
                        merged_TAD.at[last_index, "size_bin"] += row[
                            "size_bin"
                        ]
                    else:
                        left_index.append(index)
                        continue
                else:
                    merged_TAD.at[index + 1, "from.id"] = row["from.id"]
                    merged_TAD.at[index + 1, "from.coord"] = row["from.coord"]
                    merged_TAD.at[index + 1, "size_bin"] += row["size_bin"]
        else:
            left_index.append(index)
    merged_TAD = merged_TAD.loc[left_index, :]
    return merged_TAD, bins


def topdom(hic_matrix, bins, window_size, stat_filter=True, verbose=False):
    """
    Detect TADs using TopDom algorithm.

    Parameters
    ----------
    hic_matrix : band_hic_matrix or np.ndarray
        The Hi-C matrix, either as a band_hic_matrix object or a dense numpy array.
    bins : pd.DataFrame
        DataFrame containing bin information with columns 'chr', 'from.coord', 'to.coord'.
    window_size : int
        Size of the window to consider for detecting TADs.
    stat_filter : bool, optional
        Whether to apply statistical filtering to remove false positives. Default is True.
    verbose : bool, optional
        Whether to print detailed information during processing. Default is False.

    Returns
    -------
    domains : pd.DataFrame
        DataFrame containing detected TADs with columns 'chr', 'from.id', 'from.coord',
        'to.id', 'to.coord', 'tag'.
    bins : pd.DataFrame
        Updated DataFrame containing bin information with additional columns 'local.ext', 'mean.cf', 'pvalue'.
    """
    n_bins = hic_matrix.shape[0]
    mean_cf = np.zeros(n_bins)
    pvalue = np.ones(n_bins)
    local_ext = np.array(n_bins * [-0.5])
    if verbose:
        print(
            "#########################################################################"
        )
        print(
            "Step 1 : Generating binSignals by computing bin-level contact frequencies"
        )
        print(
            "#########################################################################"
        )

    ptm = time.process_time()

    # def get_mean(i,mat_data,window_size):
    #     diamond=Get_Diamond_Matrix(mat_data=hic_diag,i=i,size=window_size)
    #     return np.mean(diamond)

    for i in range(n_bins):
        diamond = Get_Diamond_Matrix(
            mat_data=hic_matrix, i=i, size=window_size
        )
        avg = np.mean(diamond)
        if avg is not np.ma.masked:
            mean_cf[i] = avg
        else:
            mean_cf[i] = 0

    eltm = time.process_time() - ptm

    if verbose:
        print("Step 1 Running Time : %f" % (eltm))
        print("Step 1 : Done !!")

        print(
            "#########################################################################"
        )
        print("Step 2 : Detect TD boundaries based on binSignals")
        print(
            "#########################################################################"
        )

    ptm = time.process_time()
    gap_idx = Which_Gap_Region2(matrix_data=hic_matrix, w=window_size, k=25)
    proc_regions = Which_Process_Region(
        rmv_idx=gap_idx, n_bins=n_bins, min_size=3
    )
    # print(gap_idx)
    # print(proc_regions)

    for i, row in proc_regions.iterrows():
        start = row["start"]
        end = row["end"]
        local_ext[start:end] = Detect_Local_Extreme(x=mean_cf[start:end])
    eltm = time.process_time()
    if verbose:
        print("Step 2 Running Time : %f" % (eltm))
        print("Step 2 : Done !!")

        print(
            "#########################################################################"
        )
        print("Step 3 : Statistical Filtering of false positive TD boundaries")
        print(
            "#########################################################################"
        )

    if stat_filter:
        ptm = time.process_time()
        if verbose:
            print("-- Matrix Scaling....")
        # scale_matrix_data = matrix_data.ravel(order='C').copy()
        # scale_matrix_data = scale_matrix_data.astype(np.float64)
        # for i in range(2*window_size):
        #     scale_matrix_data[np.arange(i,n_bins*n_bins,1+n_bins)]= scale(scale_matrix_data[np.arange(i,n_bins*n_bins,1+n_bins)])
        #     #scale_matrix_data[np.arange(i*n_bins,n_bins*n_bins,1+n_bins)]= scale(scale_matrix_data[np.arange(i*n_bins,n_bins*n_bins,1+n_bins)])
        # scale_matrix_data=scale_matrix_data.reshape(matrix_data.shape)
        scale_hic_diag = hic_matrix.copy()
        scale_hic_diag = scale_hic_diag.astype(np.float64)
        for i in range(2 * window_size):
            if isinstance(scale_hic_diag, bh.band_hic_matrix):
                diag = scale_hic_diag.diag(i)
                diag = scale_array(diag)
                scale_hic_diag.set_diag(i, diag)
            elif isinstance(scale_hic_diag, np.ndarray):
                row_idx = np.arange(n_bins - i)
                col_idx = row_idx + i
                scale_hic_diag[row_idx, col_idx] = scale_array(
                    scale_hic_diag[row_idx, col_idx]
                )
            else:
                raise ValueError(
                    "scale_hic_diag must be a band_hic_matrix or np.ndarray"
                )

        # np.savetxt('../test/scale_hic_diag.txt',scale_hic_diag.data,delimiter='\t')
        if verbose:
            print("-- Compute p-values by Wilcox Ranksum Test")
        for i, row in proc_regions.iterrows():
            start = row["start"]
            end = row["end"]
            if verbose:
                print("Process Regions from %d to %d" % (start, end))
            pvalue[start:end] = Get_Pvalue(
                scale_hic_diag,
                start,
                end,
                local_ext,
                size=window_size,
                scale=1,
            )
            # print(np.sum(pvalue[start:end]<=0.05))
            # print(pvalue)

        if verbose:
            print("-- Done!")
            print("-- Filtering False Positives")
        local_ext[
            np.intersect1d(
                np.argwhere(local_ext == -1), np.argwhere(pvalue <= 0.05)
            )
        ] = -2
        local_ext[np.argwhere(local_ext == -1)] = 0
        local_ext[np.argwhere(local_ext == -2)] = -1
        if (
            isinstance(hic_matrix, bh.band_hic_matrix)
            and hic_matrix.mask_row_col is not None
        ):
            local_ext[hic_matrix.mask_row_col] = -0.5

        if verbose:
            print("-- Done!")
            eltm = time.process_time()
            print("Step 3 Running Time : %f" % (eltm))
            print("Step 3 : Done!")

    signal_idx = np.argwhere(local_ext == -1)
    gap_idx = np.argwhere(local_ext == -0.5)
    # print(signal_idx)
    domains = Convert_Bin_To_Domain(
        bins, signal_idx, gap_idx, pvalues=pvalue, pvalue_cut=0.05
    )
    # print(domains[domains['tag']=='boundary'])
    bins["local.ext"] = local_ext
    bins["mean.cf"] = mean_cf
    bins["pvalue"] = pvalue
    return domains, bins


def scale_array(x: np.ndarray) -> np.ndarray:
    """
    Scale the input array to have zero mean and unit variance.
    Parameters
    ----------
    x : np.ndarray
        Input array to be scaled.
    Returns
    -------
    np.ndarray
        Scaled array.
    """
    if x.size == 0:
        return x
    _mean = np.mean(x)
    _std = np.std(x)

    if _std < 10 * np.finfo(_std.dtype).eps:
        _std = 1.0  # Avoid division by small standard deviation
    return (x - np.mean(x)) / (np.std(x))  # Avoid division by zero


def Get_Diamond_Matrix(mat_data, i, size):
    n_bins = mat_data.shape[0]
    if i == n_bins - 1:
        return 0
    lowerbound = max(0, i - size + 1)
    upperbound = min(i + size, n_bins - 1)

    return mat_data[lowerbound : (i + 1), (i + 1) : (upperbound + 1)]


def Which_Gap_Region2(matrix_data, w):
    n_bins = matrix_data.shape[0]
    gap = np.zeros(n_bins)
    if (
        isinstance(matrix_data, bh.band_hic_matrix)
        and matrix_data.mask_row_col is not None
    ):
        gap_idx = np.where(matrix_data.mask_row_col)[0]
        if gap_idx.size == 0:
            short_runs = []
        breaks = np.where(np.diff(gap_idx) != 1)[0] + 1
        runs = np.split(gap_idx, breaks)
        # short_runs = [(r[0], r[-1]) for r in runs if r.size < k]
        short_runs = [r for r in runs if r.size > k]
        return np.concatenate(short_runs)

    for i in range(n_bins):
        if np.sum(matrix_data[i, max(0, i - w) : min(i + w, n_bins)]) == 0:
            gap[i] = -0.5

    idx = np.argwhere(gap == -0.5).ravel()
    return idx


def Which_Process_Region(rmv_idx, n_bins, min_size=3):
    gap_idx = rmv_idx
    proc_regions = []
    proc_set = np.setdiff1d(range(n_bins), rmv_idx)
    n_proc_set = len(proc_set)

    i = 0
    while i < n_proc_set - 1:
        start = proc_set[i]
        j = i + 1
        while j < n_proc_set:
            if proc_set[j] - proc_set[j - 1] <= 1:
                j = j + 1
            else:
                proc_regions.append([start, proc_set[j - 1] + 1])
                i = j
                break
        if j >= n_proc_set - 1:
            proc_regions.append([start, proc_set[j - 1] + 1])
            break
    df_proc_regions = pd.DataFrame(proc_regions, columns=["start", "end"])
    df_proc_regions = df_proc_regions[
        abs(df_proc_regions["end"] - df_proc_regions["start"]) > min_size
    ]
    return df_proc_regions


def Detect_Local_Extreme(x):
    n_bins = len(x)
    ret = np.zeros(n_bins)
    x[np.isnan(x)] = 0

    if n_bins <= 3:
        # print(np.min(x))
        # print(np.min(y))
        ret[np.argmin(x)] = -1
        ret[np.argmax(x)] = 1
        return ret

    new_point = Data_Norm(x=np.arange(n_bins), y=x)
    x = new_point[1]

    cp = Change_Point(x=np.arange(n_bins), y=x)
    if len(cp[0]) <= 2:
        return ret
    if len(cp[0]) == n_bins:
        return ret
    for i in range(1, len(cp[0]) - 1):
        if x[cp[0][i]] >= x[cp[0][i] - 1] and x[cp[0][i]] >= x[cp[0][i] + 1]:
            ret[cp[0][i]] = 1
        elif x[cp[0][i]] < x[cp[0][i] - 1] and x[cp[0][i]] < x[cp[0][i] + 1]:
            ret[cp[0][i]] = -1

        min_val = min(x[cp[0][i - 1]], x[cp[0][i]])
        max_val = max(x[cp[0][i - 1]], x[cp[0][i]])

        if min(x[cp[0][i - 1] : cp[0][i] + 1]) < min_val:
            ret[cp[0][i - 1] + np.argmin(x[cp[0][i - 1] : cp[0][i]] + 1)] = -1
        if max(x[cp[0][i - 1] : cp[0][i] + 1]) > max_val:
            ret[cp[0][i - 1] + np.argmax(x[cp[0][i - 1] : cp[0][i]] + 1)] = 1
    return ret


def Data_Norm(x, y):
    ret_x = np.zeros(len(x))
    ret_y = np.zeros(len(y))

    ret_x[0] = x[0]
    ret_y[0] = y[0]

    diff_x = np.diff(x)
    diff_y = np.diff(y)

    epsilon = 1e-8

    scale_x = 1 / np.mean(np.abs(diff_x) + epsilon)
    scale_y = 1 / np.mean(np.abs(diff_y) + epsilon)

    for i in range(1, len(x)):
        ret_x[i] = ret_x[i - 1] + (diff_x[i - 1] * scale_x)
        ret_y[i] = ret_y[i - 1] + (diff_y[i - 1] * scale_y)

    return [ret_x, ret_y]


def Change_Point(x, y):
    if len(x) != len(y):
        print("ERROR : The length of x and y should be the same")
        return 0

    n_bins = len(x)
    Fv = np.array(n_bins * [np.nan])
    Ev = np.array(n_bins * [np.nan])
    cp = [0]

    i = 0
    Fv[0] = 0

    while i < n_bins - 1:
        j = i + 1
        Fv[j] = np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)

        while j < n_bins - 1:
            j = j + 1
            # k=list(range(i+1,j))
            Ev[j] = np.sum(
                np.abs(
                    (y[j] - y[i]) * x[i + 1 : j]
                    - (x[j] - x[i]) * y[i + 1 : j]
                    - (x[i] * y[j])
                    + (x[j] * y[i])
                )
            ) / np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2)
            Fv[j] = np.sqrt((x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2) - Ev[j]
            if (np.isnan(Fv[j])) or (np.isnan(Fv[j - 1])):
                j = j - 1
                cp.append(j)
                break
            if Fv[j] < Fv[j - 1]:
                j = j - 1
                cp.append(j)
                break
        i = j
    cp.append(n_bins - 1)

    return [cp, Fv, Ev]


def Get_Pvalue(matrix_data, start, end, local_ext, size, scale=1):
    n_bins = end - start
    pvalue = np.ones(n_bins)
    index_array = np.intersect1d(
        np.arange(start, end - 1), np.where(local_ext == -1)[0]
    )
    if index_array.shape[0]:
        for i in index_array:
            dia = Get_Diamond_Matrix2(
                matrix_data, start, end, i, size=size
            ).ravel()
            ups = Get_Upstream_Triangle(
                matrix_data, start, end, i, size=size
            ).ravel()
            downs = Get_Downstream_Triangle(
                matrix_data, start, end, i, size=size
            ).ravel()
            if (
                isinstance(matrix_data, bh.band_hic_matrix)
                and matrix_data.mask is not None
            ):
                dia = dia.compressed()
                ups = ups.compressed()
                downs = downs.compressed()

            pvalue[i - start] = stats.ranksums(
                dia * scale,
                np.append(ups, downs),
                alternative="less",
                nan_policy="omit",
            ).pvalue

    pvalue[np.isnan(pvalue)] = 1
    return pvalue


def Get_Diamond_Matrix2(mat_data, start, end, i, size):
    n_bins = end - start
    new_mat = np.array([[np.nan] * size] * size)

    for k in range(size):
        lower = min(i + 1, end - 1)
        upper = min(i + size + 1, end)
        if i - k >= start and i < end - 1:
            new_mat[size - k - 1, 0 : upper - lower] = mat_data[
                i - k, lower:upper
            ]
    return new_mat


def Get_Upstream_Triangle(mat_data, start, end, i, size):
    lower = max(start, i - size)
    tmp_mat = mat_data[lower : (i + 1), lower : (i + 1)]
    return tmp_mat[np.triu_indices(tmp_mat.shape[0], k=1)]


def Get_Downstream_Triangle(mat_data, start, end, i, size):
    if i == end - 1:
        return np.nan
    upper = min(i + size + 1, end)
    tmp_mat = mat_data[i + 1 : upper, i + 1 : upper]
    return tmp_mat[np.triu_indices(tmp_mat.shape[0], k=1)]


def Convert_Bin_To_Domain(
    bins, signal_idx, gap_idx, pvalues=None, pvalue_cut=None
):
    n_bins = len(bins)

    rmv_idx = np.setdiff1d(np.arange(n_bins), gap_idx)
    proc_region = Which_Process_Region(rmv_idx, n_bins, min_size=0)
    # print(proc_region)
    from_coord = bins.loc[proc_region["start"], "from.coord"]
    n_procs = len(proc_region)
    gap = pd.DataFrame({"from.coord": from_coord, "tag": ["gap"] * n_procs})

    rmv_idx2 = np.append(signal_idx, gap_idx)
    rmv_idx2 = np.sort(rmv_idx2)
    proc_region = Which_Process_Region(rmv_idx2, n_bins, min_size=0)
    from_coord = bins.loc[proc_region["start"], "from.coord"]
    n_procs = len(proc_region)
    # print(proc_region)
    domain = pd.DataFrame(
        {"from.coord": from_coord, "tag": ["domain"] * n_procs}
    )
    # summary=gap.append(domain)
    summary = pd.concat([gap, domain])
    # print(summary)
    rmv_idx = np.setdiff1d(np.arange(n_bins), signal_idx)
    proc_region = Which_Process_Region(rmv_idx, n_bins, min_size=1)
    n_procs = len(proc_region)
    if n_procs > 0:
        from_coord = bins.loc[proc_region["start"], "from.coord"]
        boundary = pd.DataFrame(
            {"from.coord": from_coord, "tag": ["boundary"] * n_procs}
        )
        # summary=summary.append(boundary)
        summary = pd.concat([summary, boundary])
    summary = summary.sort_values(by=["from.coord"])
    # print(summary)
    # todo
    # to_coord=summary['from.coord'].iloc[1:len(summary)].append(pd.Series([bins.loc[n_bins-1,'to.coord']]))
    to_coord = pd.concat(
        [
            summary["from.coord"].iloc[1 : len(summary)],
            pd.Series([bins.loc[n_bins - 1, "to.coord"]]),
        ]
    )
    to_coord.index = summary.index
    summary["to.coord"] = to_coord
    summary["from.id"] = summary.index
    to_id = list(summary.index)[1 : len(summary.index)]
    to_id.append(n_bins)
    to_id = pd.Series(to_id, index=summary.index)
    summary["to.id"] = to_id
    # summary['size']=summary['to.coord']-summary['from.coord']
    # summary['span']=summary['to.id']-summary['from.id']
    summary["chr"] = bins["chr"].values[0]
    summary = summary[
        ["chr", "from.id", "from.coord", "to.id", "to.coord", "tag"]
    ]
    # print(summary)
    if np.any(pvalues) and pvalue_cut != None:
        for i, row in summary.iterrows():
            if row["tag"] == "domain":
                domain_bins_idx = np.arange(row["from.id"], row["to.id"])
                p_value_constr = np.argwhere(
                    pvalues[domain_bins_idx] < pvalue_cut
                )

                if len(domain_bins_idx) == len(p_value_constr):
                    row["tag"] = "boundary"
    return summary
