# -*- coding: utf-8 -*-
"""
CPU HiCCUPS pipeline for a single chromosome (banded version).

Assumptions:
- Input is a BandHiC `band_hic_matrix` C_raw (float, unnormalized), KR vector `kr` (len=N), and distance-dependent expected vector `expected_dist`.
- Normalization: elementwise within the stored band C_norm[i,j] = C_raw[i,j] * kr[i] * kr[j], still band-stored.
- Operates directly on the band (no tiling, no margin).

Pipeline:
1) Single resolution:
   - For each pixel (i,j) compute expected eBL/eDonut/eH/eV for BL/Donut/H/V masks.
   - Log-binning on expected to get bin indices (bBL,bDonut,bH,bV).
   - Record observed = round(C_norm[i,j]) and build hist[bin, obs].
   - Estimate per-bin thresholds + FDR tables via Poisson + reverse cumulative.
   - Compute peak = observed - max(thresholds) to obtain a peak matrix.
   - Second pass: for peak>0 pixels require
       * far from diagonal (|i-j| > peak_width)
       * local maxima within peak_width window
       * valid expected values (>1e-6)
       * OE thresholds + FDR satisfied
     → generate Feature2D.
   - Centroid-cluster Feature2D.
2) Multi-resolution:
   - Run single-resolution HiCCUPS per resolution.
   - Merge via merge_all_resolutions (Juicer-like 5kb/10kb/25kb priority).
3) Output BEDPE.

Note: conceptual/testing implementation; not optimized for huge matrices.

- Only search loops where |i-j| * resolution <= max_loop_dist_bp (default 8 Mb, like CPU HiCCUPS).
- Use KR neighborhood mask (kr_neighborhood) to filter low-quality bins (mirrors Java HiCCUPS removeLowMapQFeatures).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import math

import scipy as sp
import bandhic as bh
import numpy.ma as ma
import os
import time

# Parallel/Numba imports for fast HiCCUPS pass
from joblib import Parallel, delayed
from numba import njit
import multiprocessing as mp

__all__ = [
    "hiccups",
    "load_norm_vector_from_hic",
    "load_norm_vector_from_cooler",
]


# =========================================================
# Normalization vector loaders
# =========================================================


def load_norm_vector_from_hic(hic_file, chrom, resolution, norm):
    """
    Load a normalization vector from a .hic file for a given chromosome and resolution.

    Parameters
    ----------
    hic_file : str
        Path to the .hic file.
    chrom : str
        Chromosome name, e.g., 'chr1' or '1'.
    resolution : int
        Bin size in base pairs.
    norm : str
        Normalization method, e.g., 'KR', 'VC', etc.

    Returns
    -------
    np.ndarray
        Normalization vector for the specified chromosome and resolution.

    Raises
    ------
    RuntimeError
        If the normalization vector cannot be found for the specified parameters.
    """
    try:
        import hicstraw
    except ImportError:
        raise ImportError(
            "hicstraw is required for reading .hic files. Please install it via 'pip install hic-straw'."
        )
    chrom_short = (
        chrom.replace("chr", "") if chrom.startswith("chr") else chrom
    )
    try:
        hic = hicstraw.HiCFile(hic_file)
        chrom_short_specific = (
            int(chrom_short)
            if chrom_short not in ["X", "Y", "M"]
            else chrom_short
        )
        kr = hic.getMatrixZoomData(
            chrom_short, chrom_short, "observed", norm, "BP", resolution
        ).getNormVector(chrom_short_specific)
    except Exception:
        raise RuntimeError(
            f"Normalization vector not found in .hic for {chrom} at {resolution} bp, norm={norm}"
        )
    return np.asarray(kr, dtype=float)


def load_norm_vector_from_cooler(cool_path, chrom, resolution, norm):
    """
    Load a normalization vector from a .cool or .mcool file for a given chromosome and resolution.

    Parameters
    ----------
    cool_path : str
        Path to the .cool or .mcool file.
    chrom : str
        Chromosome name, e.g., 'chr1'.
    resolution : int
        Bin size in base pairs.
    norm : str
        Normalization method, e.g., 'KR', 'VC', etc.

    Returns
    -------
    np.ndarray
        Normalization vector for the specified chromosome and resolution.

    Raises
    ------
    RuntimeError
        If the normalization vector cannot be found for the specified parameters.
    """
    import cooler

    try:
        clr = cooler.Cooler(f"{cool_path}::/resolutions/{resolution}")
        kr = clr.bins()[norm][clr.bins()["chrom"] == chrom][norm].values
    except Exception:
        raise RuntimeError(
            f"Normalization vector not found in cooler for {chrom} at {resolution} bp, norm={norm}"
        )
    return np.asarray(kr, dtype=float)


# =========================================================
# Data structures
# =========================================================


@dataclass(order=True)
class Feature2D:
    chr1: str
    start1: int
    end1: int
    chr2: str
    start2: int
    end2: int
    color: Tuple[int, int, int] = (0, 0, 0)
    attrs: Dict[str, float] = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.attrs.get(key, default)

    def set_attr(self, key: str, val: float):
        self.attrs[key] = float(val)

    def get_float(self, key: str) -> float:
        return float(self.attrs[key])

    def has_attr(self, key: str) -> bool:
        return key in self.attrs


@dataclass
class HiCCUPSConfig:
    resolution: int  # bp
    window: int = 10  # donut window (in bins)
    peak_width: int = 2  # peak half-width (in bins)
    w1: int = 40  # max bin index (for log-binning)
    fdr: float = 0.1  # FDR target (Juicer-like)
    max_count: int = 10000  # max observed tracked in hist
    cluster_radius_bp: int = 20000  # centroid merge radius (bp)
    # OE thresholds, similar to Juicer oeThreshold1/2/3
    fdrsum: float = 0.02
    oe1: float = 1.5
    oe2: float = 1.75
    oe3: float = 2.0
    max_loop_dist_bp: int = 8_000_000  # max loop search distance (bp)
    kr_neighborhood: int = (
        5  # KR neighborhood radius (bins), like Java HiCCUPS.krNeighborhood
    )
    norm: str = "KR"
    n_jobs: int = -1  # Number of parallel jobs


OBSERVED = "observed"
PEAK = "peak"
EXPECTEDBL = "expectedBL"
EXPECTEDDONUT = "expectedDonut"
EXPECTEDH = "expectedH"
EXPECTEDV = "expectedV"
BINBL = "binBL"
BINDONUT = "binDonut"
BINH = "binH"
BINV = "binV"
FDRBL = "fdrBL"
FDRDONUT = "fdrDonut"
FDRH = "fdrH"
FDRV = "fdrV"
RADIUS = "radius"
CENTROID1 = "centroid1"
CENTROID2 = "centroid2"
NUMCOLLAPSED = "numCollapsed"


def reverse_cumulative(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    if arr.ndim == 1:
        return arr[::-1].cumsum()[::-1]
    elif arr.ndim == 2:
        return np.flip(np.flip(arr, axis=axis).cumsum(axis=axis), axis=axis)
    else:
        raise ValueError("reverse_cumulative only supports 1D or 2D arrays.")


def _iter_band_indices(mat: bh.band_hic_matrix):
    """
    Iterate over valid upper-triangular banded matrix entries.

    Parameters
    ----------
    mat : band_hic_matrix
        Input banded Hi-C matrix.

    Yields
    ------
    i : int
        Row index (bin index).
    j : int
        Column index (bin index), j > i.
    k : int
        Diagonal offset (j - i).

    Notes
    -----
    Yields all valid upper-triangular (i < j) entries within the stored band, skipping masked entries.
    """
    bin_num = mat.bin_num
    diag_num = mat.diag_num
    for i in range(bin_num):
        for k in range(1, diag_num):
            j = i + k
            if j >= bin_num:
                break
            if mat.mask is not None and mat.mask[i, k]:
                continue
            yield i, j, k


# ----------------------------------------------------------
# Numba + joblib accelerated diagonal worker for HiCCUPS
# ----------------------------------------------------------

from numba import njit
import numpy as np
import math


@njit(cache=True, fastmath=True)
def _hiccups_diag_worker_numba(
    data,
    mask,
    expected_dist,
    kr,
    k_start,
    k_end,
    window,
    peak_w,
    max_loop_k,
    w1,
    max_count,
):
    """
    Efficient diagonal worker for HiCCUPS banded matrix computation.

    Parameters
    ----------
    data : np.ndarray
        Normalized Hi-C banded data array of shape (N, diag_num).
    mask : np.ndarray
        Boolean mask array of the same shape as `data`, True for masked entries.
    expected_dist : np.ndarray
        Expected values as a function of diagonal offset, length >= diag_num.
    kr : np.ndarray
        Normalization vector of length N.
    k_start : int
        Starting diagonal offset (inclusive).
    k_end : int
        Ending diagonal offset (exclusive).
    window : int
        Donut window size (in bins).
    peak_w : int
        Peak half-width (in bins).
    max_loop_k : int
        Maximum diagonal offset to consider (in bins).
    w1 : int
        Number of log-binned expected bins.
    max_count : int
        Maximum observed count tracked in histograms.

    Returns
    -------
    histBL : np.ndarray
        BL box histogram (w1, max_count+1).
    histDonut : np.ndarray
        Donut histogram (w1, max_count+1).
    histH : np.ndarray
        Horizontal crosshair histogram (w1, max_count+1).
    histV : np.ndarray
        Vertical crosshair histogram (w1, max_count+1).
    updates : list
        List of tuples per valid pixel with (i, diagDist, o, bBL, bDo, bH, bV, e_bl, e_dn, e_h, e_v).

    Notes
    -----
    This function is Numba-accelerated and processes a range of diagonal offsets for the banded matrix.
    It computes expected values for BL box, Donut, Horizontal, and Vertical crosshair masks, assigns log-binned indices,
    and accumulates observed counts and expected values for downstream thresholding and FDR.
    """
    N, diag_num = data.shape
    Nexp = len(expected_dist)
    histBL = np.zeros((w1, max_count + 1), np.int64)
    histDonut = np.zeros_like(histBL)
    histH = np.zeros_like(histBL)
    histV = np.zeros_like(histBL)
    # Parallel arrays for updates
    ii_arr = np.empty((N * (k_end - k_start),), dtype=np.int32)
    kk_arr = np.empty_like(ii_arr)
    oo_arr = np.empty_like(ii_arr)
    e_bl_arr = np.empty((N * (k_end - k_start),), dtype=np.float32)
    e_dn_arr = np.empty_like(e_bl_arr)
    e_h_arr = np.empty_like(e_bl_arr)
    e_v_arr = np.empty_like(e_bl_arr)
    bBL_arr = np.empty((N * (k_end - k_start),), dtype=np.int16)
    bDo_arr = np.empty_like(bBL_arr)
    bH_arr = np.empty_like(bBL_arr)
    bV_arr = np.empty_like(bBL_arr)
    upd_ptr = 0
    lognorm = math.log(2.0**0.33)

    def bin_val(e):
        if e <= 1.0 or math.isnan(e) or math.isinf(e):
            return 0
        idx = int(math.floor(math.log(e) / lognorm))
        if idx < 0:
            idx = 0
        if idx >= w1:
            idx = w1 - 1
        return idx

    for k in range(k_start, k_end):
        diagDist = k
        if diagDist <= peak_w:
            continue
        if diagDist >= max_loop_k:
            continue
        if diagDist >= Nexp:
            continue
        if expected_dist[diagDist] <= 0.0 or math.isnan(
            expected_dist[diagDist]
        ):
            continue
        d_diag = expected_dist[diagDist]
        for i in range(N - diagDist):
            j = i + diagDist
            if mask[i, diagDist]:
                continue
            if kr[i] == 0.0 or kr[j] == 0.0:
                continue
            # window size
            if diagDist > 1:
                wsize = min(window, (diagDist - 1) // 2)
            else:
                wsize = peak_w + 1
            if wsize <= peak_w:
                wsize = peak_w + 1
            # ----- BL box -----
            E_bl = 0.0
            Ed_bl = 0.0
            for ii in range(i + 1, min(i + wsize + 1, N)):
                for jj in range(max(j - wsize, 0), j):
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_bl += v
                                Ed_bl += expected_dist[dist]
            for ii in range(i + 1, min(i + peak_w + 1, N)):
                for jj in range(max(j - peak_w, 0), j):
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_bl -= v
                                Ed_bl -= expected_dist[dist]
            while E_bl < 16.0 and 2 * wsize < diagDist:
                E_bl = 0.0
                Ed_bl = 0.0
                wsize += 1
                for ii in range(i + 1, min(i + wsize + 1, N)):
                    for jj in range(max(j - wsize, 0), j):
                        if ii < jj:
                            dist = jj - ii
                            if dist < Nexp:
                                v = data[ii, dist]
                                if not mask[ii, dist]:
                                    E_bl += v
                                    Ed_bl += expected_dist[dist]
                                    if (
                                        i + 1 <= ii < i + peak_w + 1
                                        and j - peak_w <= jj < j
                                    ):
                                        E_bl -= v
                                        Ed_bl -= expected_dist[dist]
            # ----- Donut -----
            E_donut = 0.0
            Ed_donut = 0.0
            for ii in range(max(i - wsize, 0), min(i + wsize + 1, N)):
                for jj in range(max(j - wsize, 0), min(j + wsize + 1, N)):
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_donut += v
                                Ed_donut += expected_dist[dist]
            for ii in range(max(i - peak_w, 0), min(i + peak_w + 1, N)):
                for jj in range(max(j - peak_w, 0), min(j + peak_w + 1, N)):
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_donut -= v
                                Ed_donut -= expected_dist[dist]
            # ----- Vertical crosshair -----
            E_v = 0.0
            Ed_v = 0.0
            for ii in range(max(i - wsize, 0), max(i - peak_w, 0)):
                dist = j - ii
                v_mid = data[ii, dist]
                if not mask[ii, dist]:
                    if dist < Nexp:
                        E_donut -= v_mid
                        Ed_donut -= expected_dist[dist]
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if jj < 0 or jj >= N:
                        continue
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_v += v
                                Ed_v += expected_dist[dist]
            for ii in range(min(i + peak_w + 1, N), min(i + wsize + 1, N)):
                dist = j - ii
                v_mid = data[ii, dist]
                if not mask[ii, dist]:
                    if dist < Nexp:
                        E_donut -= v_mid
                        Ed_donut -= expected_dist[dist]
                for dj in (-1, 0, 1):
                    jj = j + dj
                    if jj < 0 or jj >= N:
                        continue
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_v += v
                                Ed_v += expected_dist[dist]
            # ----- Horizontal crosshair -----
            E_h = 0.0
            Ed_h = 0.0
            for jj in range(max(j - wsize, 0), max(j - peak_w, 0)):
                dist = jj - i
                v_mid = data[i, dist]
                if not mask[i, dist]:
                    if dist < Nexp:
                        E_donut -= v_mid
                        Ed_donut -= expected_dist[dist]
                for di in (-1, 0, 1):
                    ii = i + di
                    if ii < 0 or ii >= N:
                        continue
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_h += v
                                Ed_h += expected_dist[dist]
            for jj in range(min(j + peak_w + 1, N), min(j + wsize + 1, N)):
                dist = jj - i
                v_mid = data[i, dist]
                if not mask[i, dist]:
                    if dist < Nexp:
                        E_donut -= v_mid
                        Ed_donut -= expected_dist[dist]
                for di in (-1, 0, 1):
                    ii = i + di
                    if ii < 0 or ii >= N:
                        continue
                    if ii < jj:
                        dist = jj - ii
                        if dist < Nexp:
                            v = data[ii, dist]
                            if not mask[ii, dist]:
                                E_h += v
                                Ed_h += expected_dist[dist]
            # normalize
            e_bl = (
                (E_bl * d_diag / Ed_bl) * kr[i] * kr[j] if Ed_bl > 0 else 0.0
            )
            e_dn = (
                (E_donut * d_diag / Ed_donut) * kr[i] * kr[j]
                if Ed_donut > 0
                else 0.0
            )
            e_h = (E_h * d_diag / Ed_h) * kr[i] * kr[j] if Ed_h > 0 else 0.0
            e_v = (E_v * d_diag / Ed_v) * kr[i] * kr[j] if Ed_v > 0 else 0.0
            bBL = bin_val(e_bl)
            bDo = bin_val(e_dn)
            bH = bin_val(e_h)
            bV = bin_val(e_v)
            o = int(round(data[i, diagDist] * kr[i] * kr[j]))
            if o < 0:
                o = 0
            if o > max_count:
                o = max_count
            histBL[bBL, o] += 1
            histDonut[bDo, o] += 1
            histH[bH, o] += 1
            histV[bV, o] += 1
            ii_arr[upd_ptr] = i
            kk_arr[upd_ptr] = diagDist
            oo_arr[upd_ptr] = o
            e_bl_arr[upd_ptr] = e_bl
            e_dn_arr[upd_ptr] = e_dn
            e_h_arr[upd_ptr] = e_h
            e_v_arr[upd_ptr] = e_v
            bBL_arr[upd_ptr] = bBL
            bDo_arr[upd_ptr] = bDo
            bH_arr[upd_ptr] = bH
            bV_arr[upd_ptr] = bV
            upd_ptr += 1
    return (
        histBL,
        histDonut,
        histH,
        histV,
        upd_ptr,
        ii_arr,
        kk_arr,
        oo_arr,
        e_bl_arr,
        e_dn_arr,
        e_h_arr,
        e_v_arr,
        bBL_arr,
        bDo_arr,
        bH_arr,
        bV_arr,
    )


def compute_evalues_and_hist_parallel(
    C_norm: bh.band_hic_matrix,
    expected_dist: np.ndarray,
    conf: HiCCUPSConfig,
    kr: np.ndarray,
    n_jobs: int = -1,
    chunk_size: int = 64,
):
    """
    Compute expected values, log-binned indices, observed counts, and histograms for HiCCUPS in parallel.

    Parameters
    ----------
    C_norm : band_hic_matrix
        Normalized banded Hi-C matrix (elementwise normalized).
    expected_dist : np.ndarray
        Expected counts as a function of diagonal offset (distance dependency).
    conf : HiCCUPSConfig
        HiCCUPS configuration parameters.
    kr : np.ndarray
        Normalization vector (length N).
    n_jobs : int, optional
        Number of parallel jobs to use (default: -1, all CPUs).
    chunk_size : int, optional
        Number of diagonals per parallel worker (default: 64).

    Returns
    -------
    observed : band_hic_matrix
        Matrix of observed counts (rounded).
    eBL : band_hic_matrix
        Matrix of BL box expected values.
    eDo : band_hic_matrix
        Matrix of Donut expected values.
    eH : band_hic_matrix
        Matrix of Horizontal crosshair expected values.
    eV : band_hic_matrix
        Matrix of Vertical crosshair expected values.
    binBL : band_hic_matrix
        BL box log-binned indices.
    binDo : band_hic_matrix
        Donut log-binned indices.
    binH : band_hic_matrix
        Horizontal crosshair log-binned indices.
    binV : band_hic_matrix
        Vertical crosshair log-binned indices.
    histBL : np.ndarray
        BL box histogram (w1, max_count+1).
    histDo : np.ndarray
        Donut histogram (w1, max_count+1).
    histH : np.ndarray
        Horizontal crosshair histogram (w1, max_count+1).
    histV : np.ndarray
        Vertical crosshair histogram (w1, max_count+1).

    Notes
    -----
    This function runs the main HiCCUPS diagonal pass in parallel, using Numba-accelerated workers.
    It computes per-pixel expected values, log-binned indices, and builds histograms for downstream thresholding.
    """
    norm_data = C_norm.data
    mask = C_norm.mask
    if mask is None:
        mask = C_norm._extract_raw_mask(norm_data.shape)
    N, diag_num = norm_data.shape

    max_loop_k = conf.max_loop_dist_bp // conf.resolution

    chunks = [
        (k, min(k + chunk_size, diag_num))
        for k in range(1, diag_num, chunk_size)
    ]

    ctx = mp.get_context("fork")
    results = Parallel(
        n_jobs=n_jobs,
        backend="multiprocessing",
        prefer="processes",
    )(
        delayed(_hiccups_diag_worker_numba)(
            norm_data,
            mask,
            expected_dist,
            kr,
            k0,
            k1,
            conf.window,
            conf.peak_width,
            max_loop_k,
            conf.w1,
            conf.max_count,
        )
        for (k0, k1) in chunks
    )

    histBL = np.zeros((conf.w1, conf.max_count + 1), dtype=np.int64)
    histDo = np.zeros_like(histBL)
    histH = np.zeros_like(histBL)
    histV = np.zeros_like(histBL)

    observed = bh.zeros((N, N), diag_num=diag_num, dtype=np.int32)
    eBL = bh.zeros((N, N), diag_num=diag_num, dtype=np.float32)
    eDo = bh.zeros((N, N), diag_num=diag_num, dtype=np.float32)
    eH = bh.zeros((N, N), diag_num=diag_num, dtype=np.float32)
    eV = bh.zeros((N, N), diag_num=diag_num, dtype=np.float32)

    binBL = bh.zeros((N, N), diag_num=diag_num, dtype=np.int16)
    binDo = bh.zeros((N, N), diag_num=diag_num, dtype=np.int16)
    binH = bh.zeros((N, N), diag_num=diag_num, dtype=np.int16)
    binV = bh.zeros((N, N), diag_num=diag_num, dtype=np.int16)

    for (
        hBL,
        hDo,
        hH,
        hV,
        p,
        ii,
        kk,
        oo,
        e_bl_arr,
        e_dn_arr,
        e_h_arr,
        e_v_arr,
        bBL_arr,
        bDo_arr,
        bH_arr,
        bV_arr,
    ) in results:
        histBL += hBL
        histDo += hDo
        histH += hH
        histV += hV

        if p == 0:
            continue

        ii = ii[:p]
        kk = kk[:p]
        observed.data[ii, kk] = oo[:p]
        eBL.data[ii, kk] = e_bl_arr[:p]
        eDo.data[ii, kk] = e_dn_arr[:p]
        eH.data[ii, kk] = e_h_arr[:p]
        eV.data[ii, kk] = e_v_arr[:p]
        binBL.data[ii, kk] = bBL_arr[:p]
        binDo.data[ii, kk] = bDo_arr[:p]
        binH.data[ii, kk] = bH_arr[:p]
        binV.data[ii, kk] = bV_arr[:p]

    return (
        observed,
        eBL,
        eDo,
        eH,
        eV,
        binBL,
        binDo,
        binH,
        binV,
        histBL,
        histDo,
        histH,
        histV,
    )


# =========================================================
# Step 2: thresholds + FDR tables
# =========================================================


def compute_thresholds_and_fdr(
    hist: np.ndarray,
    conf: HiCCUPSConfig,
    pmf: np.ndarray = None,
):
    """
    Compute per-bin thresholds and FDR tables for HiCCUPS using Poisson statistics.

    Parameters
    ----------
    hist : np.ndarray
        Histogram array of shape (w1, max_count+1), where w1 is the number of log-binned expected bins.
    conf : HiCCUPSConfig
        HiCCUPS configuration object.
    pmf : np.ndarray, optional
        Precomputed Poisson PMF (not used; computed internally).

    Returns
    -------
    threshold : np.ndarray
        Array of per-bin count thresholds (length w1).
    fdrLog : np.ndarray
        Array of FDR values for each bin and observed count (shape w1, max_count+1).

    Notes
    -----
    For each log-binned expected bin, thresholds and FDRs are estimated using Poisson statistics
    and the observed histogram, following the HiCCUPS approach.
    """
    w1, width = hist.shape
    threshold = np.zeros(w1, dtype=np.float32)
    fdrLog = np.zeros_like(hist, dtype=np.float32)

    rcsHist = reverse_cumulative(hist)

    for idx in range(w1):
        cnt = rcsHist[idx, 0]
        if cnt <= 0:
            continue

        mean_obs = (hist[idx] * np.arange(width)).sum() / cnt
        if mean_obs <= 0:
            mean_obs = 1e-3

        # pmf = poisson_pmf(mean_obs, width - 1)
        pmf = sp.stats.poisson.pmf(np.arange(width), mean_obs)
        expected = rcsHist[idx, 0] * pmf
        rcsExpected = reverse_cumulative(expected)

        for j in range(width):
            if rcsExpected[j] / rcsHist[idx, j] <= conf.fdr:
                # if conf.fdr * rcsExpected[j] <= rcsHist[idx, j]:
                threshold[idx] = (width - 2) if j == 0 else (j - 1)
                break

        for j in range(width):
            s2 = rcsHist[idx, j]
            if s2 > 0:
                fdrLog[idx, j] = rcsExpected[j] / s2
            else:
                break

    return threshold, fdrLog


# =========================================================
# Step 3: Feature2D + FDR + centroid合并
# =========================================================


def generate_peak_feature(
    chr_name: str,
    res: int,
    i: int,
    j: int,
    observed: int,
    peak_val: float,
    e_bl: float,
    e_dn: float,
    e_h: float,
    e_v: float,
    b_bl: int,
    b_dn: int,
    b_h: int,
    b_v: int,
) -> Feature2D:
    """
    Generate a Feature2D object representing a candidate HiCCUPS peak.

    Parameters
    ----------
    chr_name : str
        Chromosome name.
    res : int
        Bin size in base pairs.
    i : int
        First bin index.
    j : int
        Second bin index.
    observed : int
        Observed count at (i, j).
    peak_val : float
        Peak value (observed minus threshold).
    e_bl : float
        BL box expected value.
    e_dn : float
        Donut expected value.
    e_h : float
        Horizontal crosshair expected value.
    e_v : float
        Vertical crosshair expected value.
    b_bl : int
        BL box log-binned index.
    b_dn : int
        Donut log-binned index.
    b_h : int
        Horizontal crosshair log-binned index.
    b_v : int
        Vertical crosshair log-binned index.

    Returns
    -------
    Feature2D
        Feature2D object with all relevant attributes set.
    """
    pos1 = min(i, j) * res
    pos2 = max(i, j) * res
    f = Feature2D(
        chr1=chr_name,
        start1=pos1,
        end1=pos1 + res,
        chr2=chr_name,
        start2=pos2,
        end2=pos2 + res,
        color=(0, 0, 0),
    )
    f.set_attr(OBSERVED, observed)
    f.set_attr(PEAK, peak_val)
    f.set_attr(EXPECTEDBL, e_bl)
    f.set_attr(EXPECTEDDONUT, e_dn)
    f.set_attr(EXPECTEDH, e_h)
    f.set_attr(EXPECTEDV, e_v)
    f.set_attr(BINBL, b_bl)
    f.set_attr(BINDONUT, b_dn)
    f.set_attr(BINH, b_h)
    f.set_attr(BINV, b_v)
    return f


def add_fdr_to_feature(
    f: Feature2D,
    fdrLogBL: np.ndarray,
    fdrLogDonut: np.ndarray,
    fdrLogH: np.ndarray,
    fdrLogV: np.ndarray,
):
    """
    Assign FDR values to a Feature2D object.

    Parameters
    ----------
    f : Feature2D
        Feature2D object to update.
    fdrLogBL : np.ndarray
        BL box FDR table.
    fdrLogDonut : np.ndarray
        Donut FDR table.
    fdrLogH : np.ndarray
        Horizontal crosshair FDR table.
    fdrLogV : np.ndarray
        Vertical crosshair FDR table.
    """
    obs = int(f.get_float(OBSERVED))
    bBL = int(f.get_float(BINBL))
    bDo = int(f.get_float(BINDONUT))
    bH = int(f.get_float(BINH))
    bV = int(f.get_float(BINV))

    max_obs_idx = min(obs, fdrLogBL.shape[1] - 1)
    f.set_attr(FDRBL, fdrLogBL[bBL, max_obs_idx])
    f.set_attr(FDRDONUT, fdrLogDonut[bDo, max_obs_idx])
    f.set_attr(FDRH, fdrLogH[bH, max_obs_idx])
    f.set_attr(FDRV, fdrLogV[bV, max_obs_idx])


def fdr_thresholds_satisfied(
    f: Feature2D,
    conf: HiCCUPSConfig,
) -> bool:
    """
    Evaluate whether a Feature2D passes HiCCUPS FDR and OE thresholds.

    Parameters
    ----------
    f : Feature2D
        Feature2D object with all relevant attributes.
    conf : HiCCUPSConfig
        HiCCUPS configuration parameters.

    Returns
    -------
    bool
        True if the feature passes all FDR and observed/expected thresholds, False otherwise.

    Notes
    -----
    This function implements the combined FDR and OE thresholding logic for HiCCUPS loop candidates.
    """
    obs = round(f.get_float(OBSERVED))
    expBL = f.get_float(EXPECTEDBL)
    expDn = f.get_float(EXPECTEDDONUT)
    expH = f.get_float(EXPECTEDH)
    expV = f.get_float(EXPECTEDV)
    fBL = f.get_float(FDRBL)
    fDn = f.get_float(FDRDONUT)
    fH = f.get_float(FDRH)
    fV = f.get_float(FDRV)
    numCollapsed = (
        int(f.get_float(NUMCOLLAPSED)) if f.has_attr(NUMCOLLAPSED) else 1
    )

    if min(expBL, expDn, expH, expV) <= 1e-6:
        return False

    if not (
        obs > conf.oe2 * expBL
        and obs > conf.oe2 * expDn
        and obs > conf.oe1 * expH
        and obs > conf.oe1 * expV
        and (obs > conf.oe3 * expBL or obs > conf.oe3 * expDn)
        and (numCollapsed > 1 or (fBL + fDn + fH + fV) <= conf.fdrsum)
    ):
        return False

    fdr_total = max(fBL, fDn, fH, fV)
    if fdr_total > conf.fdr:
        return False

    return True


def fdr_thresholds_satisfied(
    f: Feature2D,
    conf: HiCCUPSConfig,
) -> bool:
    """
    Evaluate whether a Feature2D passes HiCCUPS FDR and OE thresholds.

    Parameters
    ----------
    f : Feature2D
        Feature2D object with all relevant attributes.
    conf : HiCCUPSConfig
        HiCCUPS configuration parameters.

    Returns
    -------
    bool
        True if the feature passes all FDR and observed/expected thresholds, False otherwise.

    Notes
    -----
    This function implements the combined FDR and OE thresholding logic for HiCCUPS loop candidates.
    """
    obs = round(f.get_float(OBSERVED))
    expBL = f.get_float(EXPECTEDBL)
    expDn = f.get_float(EXPECTEDDONUT)
    expH = f.get_float(EXPECTEDH)
    expV = f.get_float(EXPECTEDV)
    fBL = f.get_float(FDRBL)
    fDn = f.get_float(FDRDONUT)
    fH = f.get_float(FDRH)
    fV = f.get_float(FDRV)
    numCollapsed = (
        int(f.get_float(NUMCOLLAPSED)) if f.has_attr(NUMCOLLAPSED) else 1
    )

    # if min(expBL, expDn, expH, expV) <= 1e-6:
    #     return False

    if not (
        #     obs > conf.oe2 * expBL and
        #     obs > conf.oe2 * expDn and
        #     obs > conf.oe1 * expH and
        #     obs > conf.oe1 * expV and
        #     (obs > conf.oe3 * expBL or obs > conf.oe3 * expDn) and
        (numCollapsed > 1 or (fBL + fDn + fH + fV) <= conf.fdrsum)
    ):
        return False

    # fdr_total = max(fBL, fDn, fH, fV)
    # if fdr_total > conf.fdr:
    #     return False

    return True


def coalesce_pixels_to_centroid(
    feats: List[Feature2D],
    conf: HiCCUPSConfig,
) -> List[Feature2D]:
    """
    Cluster candidate loop pixels into centroids using a fixed radius.

    Parameters
    ----------
    feats : list of Feature2D
        List of candidate loop Feature2D objects.
    conf : HiCCUPSConfig
        HiCCUPS configuration parameters.

    Returns
    -------
    merged : list of Feature2D
        List of merged centroid Feature2D objects, each with centroid coordinates and cluster attributes.

    Notes
    -----
    This function clusters pixels in the (start1, start2) plane using a fixed radius (`cluster_radius_bp`).
    The pixel with the highest observed count is used as the seed; all pixels within the radius are merged,
    and the centroid is updated iteratively.
    """
    if not feats:
        return []

    uniq = {}
    for f in feats:
        key = (f.chr1, f.start1, f.start2)
        if key not in uniq or f.get_float(OBSERVED) > uniq[key].get_float(
            OBSERVED
        ):
            uniq[key] = f
    feats = list(uniq.values())

    merged: List[Feature2D] = []
    remaining = feats[:]
    radius = conf.cluster_radius_bp

    while remaining:
        remaining.sort(key=lambda x: x.get_float(OBSERVED), reverse=True)
        seed = remaining.pop(0)
        cluster = [seed]
        cx = seed.start1
        cy = seed.start2

        changed = True
        while changed:
            changed = False
            new_remaining = []
            for f in remaining:
                dx = f.start1 - cx
                dy = f.start2 - cy
                if math.hypot(dx, dy) <= radius:
                    cluster.append(f)
                    changed = True
                else:
                    new_remaining.append(f)
            remaining = new_remaining
            if changed:
                cx = int(sum(x.start1 for x in cluster) / len(cluster))
                cy = int(sum(x.start2 for x in cluster) / len(cluster))

        seed.set_attr(NUMCOLLAPSED, len(cluster))
        seed.set_attr(CENTROID1, cx)
        seed.set_attr(CENTROID2, cy)
        rmax = 0.0
        for f in cluster:
            r = math.hypot(f.start1 - cx, f.start2 - cy)
            if r > rmax:
                rmax = r
        seed.set_attr(RADIUS, rmax)
        merged.append(seed)

    return merged


# =========================================================
# Step 4: Single-resolution HiCCUPS
# =========================================================


def run_hiccups_single_resolution(
    hic_file: str,
    chr_name: str,
    conf: HiCCUPSConfig,
):
    """
    Run the single-resolution HiCCUPS pipeline for a given chromosome.

    Parameters
    ----------
    hic_file : str
        Path to the input Hi-C file (.hic, .cool, or .mcool).
    chr_name : str
        Chromosome name (e.g., 'chr1').
    conf : HiCCUPSConfig
        HiCCUPS configuration parameters.

    Returns
    -------
    merged_loops : list of Feature2D
        List of merged loop features detected at this resolution.
    peak : band_hic_matrix
        Peak matrix (observed minus threshold) for all pixels.

    Notes
    -----
    This function runs the full single-resolution HiCCUPS pipeline:
    loads the matrix, computes normalization, expected values, thresholds, peaks, and merges loops.
    """
    res = conf.resolution
    diag = conf.max_loop_dist_bp // res
    # Load matrix, KR, expected
    t_load0 = time.perf_counter()
    print(f"Loading Hi-C matrix for {chr_name} at {res} bp resolution...")
    C_norm = bh.straw_chr(
        hic_file,
        chrom=chr_name,
        resolution=res,
        normalization=conf.norm,
        diag_num=diag,
    )
    ext = os.path.splitext(hic_file)[1].lower()
    if ext == ".hic":
        kr = load_norm_vector_from_hic(hic_file, chr_name, res, conf.norm)
    elif ext in [".cool", ".mcool"]:
        kr = load_norm_vector_from_cooler(hic_file, chr_name, res, conf.norm)
    else:
        raise ValueError(f"Unsupported Hi-C file format: {ext}")

    t_load1 = time.perf_counter()
    print(
        f"[TIME] data loading ({chr_name}, {res} bp): {t_load1 - t_load0:.3f} s"
    )

    N = C_norm.bin_num
    kr = kr[:N]
    mask_row_col = np.isnan(kr) | (kr <= 0) | (np.isinf(kr))
    C_norm.add_mask_row_col(mask_row_col=mask_row_col)
    diag_num = C_norm.diag_num
    # # Banded normalization: C_norm[i,k] = C_raw[i,k] / kr[i] / kr[i+k]
    # norm_data = np.zeros_like(C_norm.data, dtype=np.float32)
    # idx_rows = np.arange(N)
    # for k in range(diag_num):
    #     j_idx = idx_rows + k
    #     valid = j_idx < N
    #     norm_data[valid, k] = (
    #         C_raw.data[valid, k] / kr[valid] / kr[j_idx[valid]]
    #     )

    # C_norm = bh.band_hic_matrix(
    #     norm_data,
    #     diag_num=diag_num,
    #     mask=C_norm.mask,
    #     mask_row_col=C_raw.mask_row_col,
    #     band_data_input=True,
    # )
    expected_dist = C_norm.mean(axis="diag").data
    print("Computing expected values and histograms...")
    t0 = time.perf_counter()
    (
        observed,
        eBL,
        eDonut,
        eH,
        eV,
        binBL,
        binDonut,
        binH,
        binV,
        histBL,
        histDonut,
        histH,
        histV,
    ) = compute_evalues_and_hist_parallel(
        C_norm, expected_dist, conf, kr, n_jobs=conf.n_jobs
    )
    t1 = time.perf_counter()
    print(f"[TIME] compute_evalues_and_hist_parallel: {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    print("Computing thresholds and FDR tables...")
    thrBL, fdrBL = compute_thresholds_and_fdr(histBL, conf)
    thrDo, fdrDo = compute_thresholds_and_fdr(histDonut, conf)
    thrH, fdrH = compute_thresholds_and_fdr(histH, conf)
    thrV, fdrV = compute_thresholds_and_fdr(histV, conf)
    t1 = time.perf_counter()
    print(f"[TIME] thresholds + FDR: {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    print("Identifying peaks...")
    # First pass: compute peak matrix (observed minus threshold), vectorized
    peak_data = np.zeros((N, diag_num), dtype=np.float32)

    # gather bin-specific thresholds
    thr_bl_mat = thrBL[binBL.data]
    thr_do_mat = thrDo[binDonut.data]
    thr_h_mat = thrH[binH.data]
    thr_v_mat = thrV[binV.data]

    sb_mat = np.maximum.reduce(
        [
            thr_bl_mat,
            thr_do_mat,
            thr_h_mat,
            thr_v_mat,
        ]
    )

    peak_data = observed.data - sb_mat

    peak = bh.band_hic_matrix(
        peak_data,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=C_norm.mask_row_col,
        band_data_input=True,
    )
    t1 = time.perf_counter()
    print(f"[TIME] peak matrix construction: {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    print("Filtering peaks...")
    # Second pass: local maxima, FDR, and OE threshold filtering

    peak.mask[
        :, : conf.peak_width + 1
    ] = True  # mask out diagonals within peak_width
    peak.mask = np.logical_or(peak.mask, peak.data <= 0)
    peak.mask = np.logical_or(peak.mask, np.isnan(peak.data))
    peak.mask = np.logical_or(peak.mask, np.isinf(peak.data))

    peak.mask = np.logical_or(
        np.minimum.reduce([eBL.data, eDonut.data, eH.data, eV.data]) <= 1e-6,
        peak.mask,
    )

    peak.mask = np.logical_or(observed.data <= conf.oe2 * eBL.data, peak.mask)
    peak.mask = np.logical_or(
        observed.data <= conf.oe2 * eDonut.data, peak.mask
    )
    peak.mask = np.logical_or(observed.data <= conf.oe1 * eH.data, peak.mask)
    peak.mask = np.logical_or(observed.data <= conf.oe1 * eV.data, peak.mask)

    peak.mask = np.logical_or(
        np.logical_and(
            observed.data <= conf.oe3 * eBL.data,
            observed.data <= conf.oe3 * eDonut.data,
        ),
        peak.mask,
    )

    observed_clip = np.clip(observed.data, 0, conf.max_count)
    peak.mask = np.logical_or(
        np.maximum.reduce(
            [
                fdrBL[binBL.data, observed_clip],
                fdrDo[binDonut.data, observed_clip],
                fdrH[binH.data, observed_clip],
                fdrV[binV.data, observed_clip],
            ]
        )
        > conf.fdr,
        peak.mask,
    )
    del observed_clip

    candidates: List[Feature2D] = []

    rows, cols = np.where(~peak.mask)
    for i, k in zip(rows, cols):
        diagDist = k
        j = i + diagDist
        val = peak.data[i, k]
        # # Local maxima check (only compare within banded region)
        # pw = conf.peak_width
        # max_val = -np.inf
        # for ii in range(max(0, i - pw), min(N, i + pw + 1)):
        #     for jj in range(max(0, j - pw), min(N, j + pw + 1)):
        #         vv = peak[ii, jj]
        #         if ma.is_masked(vv):
        #             continue
        #         if vv > max_val:
        #             max_val = vv
        # if val < max_val:
        #     continue

        e_bl = eBL.data[i, k]
        e_dn = eDonut.data[i, k]
        e_hh = eH.data[i, k]
        e_vv = eV.data[i, k]

        o = observed.data[i, k]
        bBL = int(binBL.data[i, k])
        bDo = int(binDonut.data[i, k])
        bH = int(binH.data[i, k])
        bV = int(binV.data[i, k])

        f = generate_peak_feature(
            chr_name,
            conf.resolution,
            i,
            j,
            o,
            val,
            e_bl,
            e_dn,
            e_hh,
            e_vv,
            bBL,
            bDo,
            bH,
            bV,
        )

        add_fdr_to_feature(f, fdrBL, fdrDo, fdrH, fdrV)
        candidates.append(f)

    t1 = time.perf_counter()
    print(f"[TIME] local maxima + filtering: {t1 - t0:.3f} s")

    t0 = time.perf_counter()
    merged_loops = coalesce_pixels_to_centroid(candidates, conf)
    filtered_loops = []
    for loop in merged_loops:
        if fdr_thresholds_satisfied(loop, conf):
            filtered_loops.append(loop)
    t1 = time.perf_counter()
    print(f"[TIME] centroid merging: {t1 - t0:.3f} s")
    # merged_loops = candidates
    return filtered_loops, peak


# =========================================================
# Step 5: multi-resolution HiCCUPS + merging
# =========================================================


def euclid_bp(f1: Feature2D, f2: Feature2D) -> float:
    dx = f1.start1 - f2.start1
    dy = f1.start2 - f2.start2
    return math.hypot(dx, dy)


def extract_reproducible_centroids(
    list_a: List[Feature2D], list_b: List[Feature2D], radius_bp: int
) -> List[Feature2D]:
    centroids = []
    for fb in list_b:
        for fa in list_a:
            if fa.chr1 != fb.chr1:
                continue
            if euclid_bp(fa, fb) <= radius_bp:
                centroids.append(fb)
                break
    return centroids


def extract_peaks_near_centroids(
    peaks: List[Feature2D], centroids: List[Feature2D], radius_bp: int
) -> List[Feature2D]:
    out = []
    for f in peaks:
        for c in centroids:
            if f.chr1 == c.chr1 and euclid_bp(f, c) <= radius_bp:
                out.append(f)
                break
    return out


def extract_peaks_not_near_centroids(
    peaks: List[Feature2D], centroids: List[Feature2D], radius_bp: int
) -> List[Feature2D]:
    out = []
    for f in peaks:
        keep = True
        for c in centroids:
            if f.chr1 == c.chr1 and euclid_bp(f, c) <= radius_bp:
                keep = False
                break
        if keep:
            out.append(f)
    return out


def get_peaks_near_diagonal(
    peaks: List[Feature2D], max_dist_bp: int
) -> List[Feature2D]:
    out = []
    for f in peaks:
        if abs(f.start2 - f.start1) <= max_dist_bp:
            out.append(f)
    return out


def get_strong_peaks(
    peaks: List[Feature2D], min_observed: float
) -> List[Feature2D]:
    out = []
    for f in peaks:
        if f.get_float(OBSERVED) >= min_observed:
            out.append(f)
    return out


def remove_duplicates(features: List[Feature2D]) -> List[Feature2D]:
    best: Dict[Tuple[str, int, int], Feature2D] = {}
    for f in features:
        key = (f.chr1, f.start1, f.start2)
        if key not in best or f.get_float(PEAK) > best[key].get_float(PEAK):
            best[key] = f
    return list(best.values())


def merge_all_resolutions(
    looplists: Dict[int, List[Feature2D]]
) -> List[Feature2D]:
    """
    Mimics Juicer's HiCCUPSUtils.mergeAllResolutions logic:

    - If 5 kb and/or 10 kb resolutions are available:
        * If both are present:
            - Merge 5 kb and 10 kb loops within overlapping centroid regions
            - Add 10 kb peaks that do not have nearby 5 kb centroids
            - Add 5 kb peaks that are either close to the diagonal or have strong signal
        * If only 5 kb or only 10 kb is present, directly use that resolution
    - If 25 kb resolution is available:
        * If a merged list already exists:
            - Add 25 kb peaks that are far from existing merged centroids
        * Otherwise, set merged = 25 kb peaks
    - If none of 5 kb / 10 kb / 25 kb resolutions are used:
        * Simply take the union of all resolutions and remove duplicates
    """
    merged: List[Feature2D] = []
    list_altered = False

    has5 = 5000 in looplists and len(looplists[5000]) > 0
    has10 = 10000 in looplists and len(looplists[10000]) > 0
    has25 = 25000 in looplists and len(looplists[25000]) > 0

    if has5 or has10:
        if has5 and has10:
            five = looplists[5000]
            ten = looplists[10000]

            c5 = extract_reproducible_centroids(ten, five, 2 * 10000)
            merged_tmp = extract_peaks_near_centroids(five, c5, 2 * 10000)

            c10 = extract_reproducible_centroids(five, ten, 2 * 10000)
            distant10 = extract_peaks_not_near_centroids(ten, c10, 2 * 10000)
            merged_tmp.extend(distant10)

            near_diag = get_peaks_near_diagonal(five, 110000)
            strong = get_strong_peaks(five, 100)

            merged_tmp.extend(near_diag)
            merged_tmp.extend(strong)

            merged = remove_duplicates(merged_tmp)
        elif has5:
            merged = remove_duplicates(looplists[5000])
        else:
            merged = remove_duplicates(looplists[10000])

        list_altered = True

    if has25:
        twenty5 = looplists[25000]
        if list_altered:
            c25 = extract_reproducible_centroids(merged, twenty5, 2 * 25000)
            distant25 = extract_peaks_not_near_centroids(
                twenty5, c25, 2 * 25000
            )
            merged.extend(distant25)
            merged = remove_duplicates(merged)
        else:
            merged = remove_duplicates(twenty5)
        list_altered = True

    # 若 5/10/25 都没用上：合并所有分辨率
    if not list_altered:
        tmp: List[Feature2D] = []
        for lst in looplists.values():
            tmp.extend(lst)
        merged = remove_duplicates(tmp)

    return merged


def run_hiccups_multi_resolution(
    hic_file: str,
    chr_name: str,
    configs: Dict[int, HiCCUPSConfig],
):
    """
    Multi-resolution HiCCUPS:
    - Run run_hiccups_single_resolution for each resolution
    - Merge results using merge_all_resolutions
    """
    looplists: Dict[int, List[Feature2D]] = {}
    for res, conf in configs.items():
        loops, _ = run_hiccups_single_resolution(
            hic_file,
            chr_name,
            conf,
        )
        looplists[res] = loops
    merged = merge_all_resolutions(looplists)
    return merged


# -------------------------------------------------------------
# High-level HiCCUPS function API (clean & structured)
# -------------------------------------------------------------


def _default_peak_width(res):
    """Juicer default peak width."""
    if res == 1000:
        return 20
    if res == 5000:
        return 4
    if res == 10000:
        return 2
    if res == 25000:
        return 1
    return 3


def _default_window(res):
    """Juicer default donut window."""
    if res == 1000:
        return 35
    if res == 5000:
        return 7
    if res == 10000:
        return 5
    if res == 25000:
        return 3
    return 5


from typing import Union


def hiccups(
    hic_file: str,
    chroms: Union[str, list],
    resolutions: list,
    fdr: list = None,
    peak_width: list = None,
    window: list = None,
    thresholds: list = (0.02, 1.5, 1.75, 2.0),
    centroid_radius: list = None,
    max_loop_dist_bp: int = 2_000_000,
    kr_neighborhood: int = 5,
    matrix_size: int = 512,
    norm: str = "KR",
    out_path: str = None,
    n_jobs: int = -1,
):
    """
    Run the HiCCUPS loop-calling algorithm on Hi-C data using the BandHiC data structure.

    This function provides a high-level interface to a BandHiC-based reimplementation
    of the HiCCUPS algorithm. It supports single- or multi-resolution loop detection
    across one or more chromosomes and internally applies multiprocessing to accelerate
    computation on large, high-resolution Hi-C matrices.

    Parameters
    ----------
    hic_file : str
        Path to the input Hi-C file. Supported formats include ``.hic``, ``.cool``,
        and ``.mcool``.
    chroms : str or list of str
        Chromosome name or list of chromosome names (e.g., ``"chr1"`` or
        ``["chr1", "chr2"]``).
    resolutions : list of int
        List of bin resolutions (in base pairs) at which HiCCUPS will be executed.
    fdr : list of float or float, optional
        Target false discovery rate (FDR) for loop detection at each resolution.
        If a single value is provided, it is applied to all resolutions.
    peak_width : list of int or int, optional
        Peak half-width (in bins) for each resolution. Defaults follow Juicer
        recommendations if not specified.
    window : list of int or int, optional
        Donut window size (in bins) for each resolution. Defaults follow Juicer
        recommendations if not specified.
    thresholds : list of float, optional
        Threshold parameters ``[fdrsum, oe1, oe2, oe3]`` used for HiCCUPS filtering
        and cross-resolution merging.
    centroid_radius : list of int or int, optional
        Radius (in base pairs) for clustering nearby loop pixels into centroids
        at each resolution.
    max_loop_dist_bp : int, optional
        Maximum genomic distance (in base pairs) for loop search. Default is
        2,000,000 bp.
    kr_neighborhood : int, optional
        Radius (in bins) used to mask low-quality bins based on the KR
        normalization vector.
    matrix_size : int, optional
        Reserved parameter for compatibility; not used in the current implementation.
    norm : str, optional
        Normalization method to use (e.g., ``"KR"``, ``"VC"``). Default is ``"KR"``.
    out_path : str, optional
        Path to write the output loops in BEDPE format. If ``None``, results
        are not written to disk.
    n_jobs : int, optional
        Number of parallel worker processes to use. ``-1`` uses all available CPUs.

    Returns
    -------
    loops : list of Feature2D
        List of detected chromatin loops after multi-resolution merging.
    """
    num_res = len(resolutions)

    def _expand_param(param, default_fn=None, default_val=None):
        if param is None:
            if default_fn is not None:
                return [default_fn(r) for r in resolutions]
            elif default_val is not None:
                return [default_val] * num_res
        if isinstance(param, (int, float)):
            return [param] * num_res
        if isinstance(param, list):
            if len(param) == 1:
                return param * num_res
            if len(param) != num_res:
                raise ValueError(f"Parameter length mismatch for {param}")
            return param
        raise ValueError(f"Unsupported parameter type: {param}")

    fdr = _expand_param(fdr, default_val=0.1)
    peak_width = _expand_param(peak_width, default_fn=_default_peak_width)
    window = _expand_param(window, default_fn=_default_window)
    centroid_radius = _expand_param(centroid_radius, default_val=20000)
    if isinstance(thresholds, (int, float)):
        thresholds = [thresholds] * 4
    if len(thresholds) != 4:
        raise ValueError("thresholds must have length 4 or be a scalar.")
    fdrsum, oe1, oe2, oe3 = thresholds
    # Build configs
    configs_by_res: Dict[int, HiCCUPSConfig] = {}
    for idx, res in enumerate(resolutions):
        conf = HiCCUPSConfig(
            resolution=res,
            window=window[idx],
            peak_width=peak_width[idx],
            fdr=fdr[idx],
            oe1=oe1,
            oe2=oe2,
            oe3=oe3,
            fdrsum=fdrsum,
            cluster_radius_bp=centroid_radius[idx],
            max_loop_dist_bp=max_loop_dist_bp,
            kr_neighborhood=kr_neighborhood,
            norm=norm,
            n_jobs=n_jobs,
        )
        configs_by_res[res] = conf

    # Support chroms as str or list
    if isinstance(chroms, str):
        chrom_list = [chroms]
    else:
        chrom_list = list(chroms)

    all_loops = []
    for chrom in chrom_list:
        loops_chr = run_hiccups_multi_resolution(
            hic_file,
            chrom,
            configs_by_res,
        )
        all_loops.extend(loops_chr)

    if out_path is not None:
        write_bedpe(all_loops, out_path)
    return all_loops


# =========================================================
# Step 6: BEDPE Output
# =========================================================


def write_bedpe(loops: List[Feature2D], path: str):
    """
    header:
    #chr1 x1 x2 chr2 y1 y2 name score strand1 strand2 color observed expectedBL expectedDonut expectedH expectedV fdrBL fdrDonut fdrH fdrV numCollapsed centroid1 centroid2 radius
    """
    with open(path, "w") as f:
        header = [
            "#chr1",
            "x1",
            "x2",
            "chr2",
            "y1",
            "y2",
            "name",
            "score",
            "strand1",
            "strand2",
            "color",
            "observed",
            "expectedBL",
            "expectedDonut",
            "expectedH",
            "expectedV",
            "fdrBL",
            "fdrDonut",
            "fdrH",
            "fdrV",
            "numCollapsed",
            "centroid1",
            "centroid2",
            "radius",
        ]
        f.write("\t".join(header) + "\n")
        for feat in loops:
            score = feat.get_float(PEAK)
            observed = (
                feat.get_float(OBSERVED)
                if OBSERVED in feat.attrs
                else float("nan")
            )
            expBL = (
                feat.get_float(EXPECTEDBL)
                if EXPECTEDBL in feat.attrs
                else float("nan")
            )
            expDonut = (
                feat.get_float(EXPECTEDDONUT)
                if EXPECTEDDONUT in feat.attrs
                else float("nan")
            )
            expH = (
                feat.get_float(EXPECTEDH)
                if EXPECTEDH in feat.attrs
                else float("nan")
            )
            expV = (
                feat.get_float(EXPECTEDV)
                if EXPECTEDV in feat.attrs
                else float("nan")
            )

            fdrBL_val = (
                feat.get_float(FDRBL) if FDRBL in feat.attrs else float("nan")
            )
            fdrDo_val = (
                feat.get_float(FDRDONUT)
                if FDRDONUT in feat.attrs
                else float("nan")
            )
            fdrH_val = (
                feat.get_float(FDRH) if FDRH in feat.attrs else float("nan")
            )
            fdrV_val = (
                feat.get_float(FDRV) if FDRV in feat.attrs else float("nan")
            )

            numCollapsed_val = (
                feat.get_float(NUMCOLLAPSED)
                if NUMCOLLAPSED in feat.attrs
                else float("nan")
            )
            centroid1_val = (
                feat.get_float(CENTROID1)
                if CENTROID1 in feat.attrs
                else float("nan")
            )
            centroid2_val = (
                feat.get_float(CENTROID2)
                if CENTROID2 in feat.attrs
                else float("nan")
            )
            radius_val = (
                feat.get_float(RADIUS)
                if RADIUS in feat.attrs
                else float("nan")
            )

            color_str = ",".join(map(str, feat.color)) if feat.color else ""

            line_items = [
                feat.chr1,
                str(feat.start1),
                str(feat.end1),
                feat.chr2,
                str(feat.start2),
                str(feat.end2),
                ".",  # name
                f"{score:.4g}",
                ".",  # strand1
                ".",  # strand2
                color_str,
                f"{observed:.6g}",
                f"{expBL:.6g}",
                f"{expDonut:.6g}",
                f"{expH:.6g}",
                f"{expV:.6g}",
                f"{fdrBL_val:.6g}",
                f"{fdrDo_val:.6g}",
                f"{fdrH_val:.6g}",
                f"{fdrV_val:.6g}",
                f"{numCollapsed_val:.6g}",
                f"{centroid1_val:.6g}",
                f"{centroid2_val:.6g}",
                f"{radius_val:.6g}",
            ]
            f.write("\t".join(line_items) + "\n")


# =========================================================
# Simple test function for HiCCUPS
# =========================================================
def test_hiccups_basic():
    """
    Minimal smoke test for HiCCUPS:
      - Runs hiccups on a small chromosome with one resolution.
      - Does not validate biological correctness; only checks that
        the pipeline executes end‑to‑end without errors.
    """
    # hic_path = "/Users/wwb/workspace-local/call_loop/data/GSE63525_GM12878_insitu_primary_replicate_combined.hic"
    hic_path = "/Users/wwb/workspace-local/call_loop/data/GSE130275_mESC_WT_combined_1.3B_microc.hic"

    chrom = "chr19"
    resolutions = [1000]

    loops = hiccups(
        hic_file=hic_path,
        chroms=chrom,
        resolutions=resolutions,
        out_path="../../test/hiccups_mESC_output_1000_chr19.bedpe",
        n_jobs=-1,
    )
    print(f"Test completed. Loops detected: {len(loops)}")


if __name__ == "__main__":
    import time, os, psutil

    proc = psutil.Process(os.getpid())

    mem_before = proc.memory_info().rss / 1024**2
    t0 = time.perf_counter()

    test_hiccups_basic()

    t1 = time.perf_counter()
    mem_after = proc.memory_info().rss / 1024**2
    mem_peak = (
        proc.memory_info().peak_wset / 1024**2
        if hasattr(proc.memory_info(), "peak_wset")
        else mem_after
    )

    print(f"[HiCCUPS] time = {t1 - t0:.2f} s")
    print(f"[HiCCUPS] RSS before = {mem_before:.1f} MiB")
    print(f"[HiCCUPS] RSS after  = {mem_after:.1f} MiB")
