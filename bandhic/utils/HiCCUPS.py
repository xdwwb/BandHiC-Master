# -*- coding: utf-8 -*-
"""
CPU HiCCUPS pipeline for a single chromosome (banded version).

假设：
- 输入是 BandHiC 的 band_hic_matrix 对象 C_raw (float, 未KR)、KR向量 kr（len=N）、距离依赖的 expected 向量 expected_dist。
- 归一化方式：逐条带宽内元素相乘 C_norm[i,j] = C_raw[i,j]*kr[i]*kr[j]，仍以带状存储。
- 只在带宽范围内直接计算（不做分块，不做 margin）。

Pipeline:
1) 单分辨率：
   - 对每个像素(i,j)计算 BL/Donut/H/V 掩膜对应的期望 eBL/eDonut/eH/eV。
   - 按 e 的 log-binning 计算 bin 索引 (bBL,bDonut,bH,bV)。
   - 记录 observed = round(C_norm[i,j])，构建 hist[bin, obs]。
   - 用 Poisson + 反累积分布估计各 bin 的阈值 + FDR 表。
   - 按像素计算 peak = observed - max(thresholds)，得到 peak 矩阵。
   - 第二遍扫描：对 peak>0 的像素，要求：
       * 远离对角线 (|i-j| > peak_width)
       * local maxima（在 peakWidth 邻域内是最大值）
       * expected 都有效 (>1e-6)
       * OE 阈值 + FDR 条件都满足
     → 生成 Feature2D。
   - 对 Feature2D 做 centroid 聚类合并。
2) 多分辨率：
   - 对每个 resolution 跑一遍单分辨率 HiCCUPS。
   - 用 merge_all_resolutions 规则合并（仿 Juicer: 5kb/10kb/25kb 优先）。
3) 输出 BEDPE。

注意：这是概念/测试实现，性能不适合超大矩阵。

- 只在 |i-j| * resolution <= max_loop_dist_bp 的范围内搜索 loops（默认 8 Mb，仿 CPU HiCCUPS）。
- 使用 KR 邻域掩膜（kr_neighborhood）过滤低质量 bin（仿 Java HiCCUPS 的 removeLowMapQFeatures）。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import math
import bandhic as bh
import numpy.ma as ma
import os

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


@dataclass
class HiCCUPSConfig:
    resolution: int               # bp
    window: int = 10              # donut window (in bins)
    peak_width: int = 2           # peak half-width (in bins)
    w1: int = 300                 # max bin index (for log-binning)
    fdr: float = 0.1              # FDR 目标（与 Juicer 相似）
    max_count: int = 1000         # hist 中追踪的最大 observed
    cluster_radius_bp: int = 20000  # centroid 合并半径（bp）
    # OE thresholds，仿 Juicer 的 oeThreshold1/2/3
    oe1: float = 1.5
    oe2: float = 2.0
    oe3: float = 3.0
    # 最大 loop 搜索距离（单位：bp），CPU HiCCUPS 默认 8 Mb 附近
    max_loop_dist_bp: int = 8_000_000
    # KR 邻域半径（bin），用于模拟 Java HiCCUPS.krNeighborhood 掩膜
    kr_neighborhood: int = 2


# 属性键（参考 Juicer Translation）
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


# =========================================================
# Utility: Poisson, reverse-cumulative
# =========================================================

def poisson_pmf(lmbda: float, max_k: int) -> np.ndarray:
    """Poisson(λ) PMF for k=0..max_k; 递推实现 + 归一化。"""
    pmf = np.zeros(max_k + 1, dtype=np.float64)
    pmf[0] = math.exp(-lmbda)
    for k in range(1, max_k + 1):
        pmf[k] = pmf[k - 1] * lmbda / k
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def reverse_cumulative(arr: np.ndarray) -> np.ndarray:
    """f[k] -> g[k] = sum_{x>=k} f[x]."""
    return arr[::-1].cumsum()[::-1]


def _iter_band_indices(mat: bh.band_hic_matrix):
    """
    Iterate over valid upper-triangular band positions (i<j, in-band).
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


# =========================================================
# Step 1: compute per-pixel expected / bins / observed / hist
# =========================================================

def compute_evalues_and_hist(
    C_norm: bh.band_hic_matrix,
    expected_dist: np.ndarray,
    conf: HiCCUPSConfig,
    kr: np.ndarray,
):
    """
    第一遍扫描：
      - 对每个像素 (i,j) (只看上三角 i<j) 计算 eBL/eDonut/eH/eV，
        bin 索引、observed，并构建 hist[bin,obs]。
    """
    N = C_norm.bin_num
    diag_num = C_norm.diag_num
    window = conf.window
    peak_w = conf.peak_width
    w1 = conf.w1
    max_count = conf.max_count
    
        # KR 掩膜：仿 HiCCUPSUtils.removeLowMapQFeatures 中 nearbyValuesClear
    kr_arr = np.asarray(kr, dtype=float)
    kr_nan = np.isnan(kr_arr) | (kr_arr == 0)
    valid_kr = np.ones(N, dtype=bool)
    r = conf.kr_neighborhood
    if r > 0:
        for i in range(N):
            i0 = max(0, i - r)
            i1 = min(N, i + r + 1)
            if np.any(kr_nan[i0:i1]):
                valid_kr[i] = False

    observed = bh.zeros((N, diag_num), dtype=np.int32)
    observed.add_mask_row_col(~valid_kr)
    eBL = bh.zeros((N, diag_num), dtype=np.float32)
    eBL.add_mask_row_col(~valid_kr)
    eDonut = bh.zeros((N, diag_num), dtype=np.float32)
    eDonut.add_mask_row_col(valid_kr)
    eH = bh.zeros((N, diag_num), dtype=np.float32)
    eH.add_mask_row_col(~valid_kr)
    eV = bh.zeros((N, diag_num), dtype=np.float32)
    eV.add_mask_row_col(~valid_kr)
    binBL = bh.zeros((N, diag_num), dtype=np.int32)
    binBL.add_mask_row_col(~valid_kr)
    binDonut = bh.zeros((N, diag_num), dtype=np.int32)
    binDonut.add_mask_row_col(~valid_kr)
    binH = bh.zeros((N, diag_num), dtype=np.int32)
    binH.add_mask_row_col(~valid_kr)
    binV = bh.zeros((N, diag_num), dtype=np.int32)
    binV.add_mask_row_col(~valid_kr)

    histBL = np.zeros((w1, max_count + 1), dtype=np.int64)
    histDonut = np.zeros((w1, max_count + 1), dtype=np.int64)
    histH = np.zeros((w1, max_count + 1), dtype=np.int64)
    histV = np.zeros((w1, max_count + 1), dtype=np.int64)

    lognorm = math.log(2.0 ** 0.33)
    Nexp = len(expected_dist)

    def bin_val(e: float) -> int:
        if e <= 1 or math.isnan(e) or math.isinf(e):
            return 0
        idx = int(math.floor(math.log(e) / lognorm))
        if idx < 0:
            idx = 0
        if idx >= w1:
            idx = w1 - 1
        return idx

    valid_mask = np.zeros((N, diag_num), dtype=bool)

    for t_row, t_col, offset in _iter_band_indices(C_norm):
        diagDist = offset
        if diagDist <= peak_w:
            continue
        if not (valid_kr[t_row] and valid_kr[t_col]):
                    continue
        if diagDist * conf.resolution > conf.max_loop_dist_bp:
            continue
        if diagDist >= Nexp:
            continue
        if expected_dist[diagDist] <= 0 or np.isnan(expected_dist[diagDist]):
            continue

        d_diag = expected_dist[diagDist]

        if diagDist > 1:
            wsize = min(window, (diagDist - 1) // 2)
        else:
            wsize = peak_w + 1
        if wsize <= peak_w:
            wsize = peak_w + 1

        # ---------------- BL box ----------------
        E_bl = 0.0
        Ed_bl = 0.0
        for i in range(t_row + 1, min(t_row + wsize + 1, N)):
            for j in range(max(t_col - wsize, 0), t_col):
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_bl += v
                    Ed_bl += expected_dist[dist]

        for i in range(t_row + 1, min(t_row + peak_w + 1, N)):
            for j in range(max(t_col - peak_w, 0), t_col):
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_bl -= v
                    Ed_bl -= expected_dist[dist]

        while E_bl < 16 and 2 * wsize < diagDist:
            E_bl = 0.0
            Ed_bl = 0.0
            wsize += 1
            for i in range(t_row + 1, min(t_row + wsize + 1, N)):
                for j in range(max(t_col - wsize, 0), t_col):
                    v = C_norm[i, j]
                    if ma.is_masked(v):
                        continue
                    dist = j - i
                    if dist < Nexp:
                        E_bl += v
                        Ed_bl += expected_dist[dist]
                        if (t_row + 1 <= i < t_row + peak_w + 1 and
                            t_col - peak_w <= j < t_col):
                            E_bl -= v
                            Ed_bl -= expected_dist[dist]

        # ---------------- Donut ----------------
        E_donut = 0.0
        Ed_donut = 0.0
        for i in range(max(t_row - wsize, 0), min(t_row + wsize + 1, N)):
            for j in range(max(t_col - wsize, 0), min(t_col + wsize + 1, N)):
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                if i < j:
                    dist = j - i
                    if dist < Nexp:
                        E_donut += v
                        Ed_donut += expected_dist[dist]

        for i in range(max(t_row - peak_w, 0), min(t_row + peak_w + 1, N)):
            for j in range(max(t_col - peak_w, 0), min(t_col + peak_w + 1, N)):
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                if i < j:
                    dist = j - i
                    if dist < Nexp:
                        E_donut -= v
                        Ed_donut -= expected_dist[dist]

        # ---------------- Vertical crosshair ----------------
        E_v = 0.0
        Ed_v = 0.0
        for i in range(max(t_row - wsize, 0), max(t_row - peak_w, 0)):
            v_mid = C_norm[i, t_col]
            if not ma.is_masked(v_mid):
                dist = abs(i - t_col)
                if dist < Nexp:
                    E_donut -= v_mid
                    Ed_donut -= expected_dist[dist]
            for dj in (-1, 0, 1):
                j = t_col + dj
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_v += v
                    Ed_v += expected_dist[dist]

        for i in range(min(t_row + peak_w + 1, N), min(t_row + wsize + 1, N)):
            v_mid = C_norm[i, t_col]
            if not ma.is_masked(v_mid):
                dist = abs(i - t_col)
                if dist < Nexp:
                    E_donut -= v_mid
                    Ed_donut -= expected_dist[dist]
            for dj in (-1, 0, 1):
                j = t_col + dj
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_v += v
                    Ed_v += expected_dist[dist]

        # ---------------- Horizontal crosshair ----------------
        E_h = 0.0
        Ed_h = 0.0
        for j in range(max(t_col - wsize, 0), max(t_col - peak_w, 0)):
            v_mid = C_norm[t_row, j]
            if not ma.is_masked(v_mid):
                dist = abs(t_row - j)
                if dist < Nexp:
                    E_donut -= v_mid
                    Ed_donut -= expected_dist[dist]
            for di in (-1, 0, 1):
                i = t_row + di
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_h += v
                    Ed_h += expected_dist[dist]

        for j in range(min(t_col + peak_w + 1, N), min(t_col + wsize + 1, N)):
            v_mid = C_norm[t_row, j]
            if not ma.is_masked(v_mid):
                dist = abs(t_row - j)
                if dist < Nexp:
                    E_donut -= v_mid
                    Ed_donut -= expected_dist[dist]
            for di in (-1, 0, 1):
                i = t_row + di
                v = C_norm[i, j]
                if ma.is_masked(v):
                    continue
                dist = j - i
                if dist < Nexp:
                    E_h += v
                    Ed_h += expected_dist[dist]

        def safe_e(E, Ed):
            if Ed <= 0:
                return 0.0
            return (E * d_diag) / Ed

        e_bl = safe_e(E_bl, Ed_bl)
        e_dn = safe_e(E_donut, Ed_donut)
        e_hh = safe_e(E_h, Ed_h)
        e_vv = safe_e(E_v, Ed_v)

        eBL[t_row, t_col] = e_bl
        eDonut[t_row, t_col] = e_dn
        eH[t_row, t_col] = e_hh
        eV[t_row, t_col] = e_vv

        bBL = bin_val(e_bl)
        bDo = bin_val(e_dn)
        bH = bin_val(e_hh)
        bV = bin_val(e_vv)
        binBL[t_row, t_col] = bBL
        binDonut[t_row, t_col] = bDo
        binH[t_row, t_col] = bH
        binV[t_row, t_col] = bV

        o = int(round(C_norm.data[t_row, offset]))
        if o < 0:
            o = 0
        if o > max_count:
            o = max_count
        observed[t_row, t_col] = o

        histBL[bBL, o] += 1
        histDonut[bDo, o] += 1
        histH[bH, o] += 1
        histV[bV, o] += 1
        valid_mask[t_row, offset] = True

    return (observed,
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
            valid_mask,
            row_col_mask,
            )


# =========================================================
# Step 2: thresholds + FDR tables
# =========================================================

def compute_thresholds_and_fdr(
    hist: np.ndarray,
    conf: HiCCUPSConfig,
):
    """
    hist: (w1, max_count+1)
    返回:
      threshold: (w1,)
      fdrLog: (w1, max_count+1)
    """
    w1, width = hist.shape
    threshold = np.zeros(w1, dtype=np.float32)
    fdrLog = np.zeros_like(hist, dtype=np.float32)

    rcsHist = reverse_cumulative(hist)

    for idx in range(w1):
        if rcsHist[idx, 0] <= 0:
            continue

        cnt = hist[idx].sum()
        if cnt <= 0:
            continue

        mean_obs = (hist[idx] * np.arange(width)).sum() / cnt
        if mean_obs <= 0:
            mean_obs = 1e-3

        pmf = poisson_pmf(mean_obs, width - 1)
        expected = rcsHist[idx, 0] * pmf
        rcsExpected = reverse_cumulative(expected)

        for j in range(width):
            if conf.fdr * rcsExpected[j] <= rcsHist[idx, j]:
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
    fdrLogV: np.ndarray
):
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
    obs = round(f.get_float(OBSERVED))
    expBL = f.get_float(EXPECTEDBL)
    expDn = f.get_float(EXPECTEDDONUT)
    expH = f.get_float(EXPECTEDH)
    expV = f.get_float(EXPECTEDV)
    fBL = f.get_float(FDRBL)
    fDn = f.get_float(FDRDONUT)
    fH = f.get_float(FDRH)
    fV = f.get_float(FDRV)

    if min(expBL, expDn, expH, expV) <= 1e-6:
        return False

    if not (
        obs > conf.oe2 * expBL and
        obs > conf.oe2 * expDn and
        obs > conf.oe1 * expH and
        obs > conf.oe1 * expV and
        (obs > conf.oe3 * expBL or obs > conf.oe3 * expDn)
    ):
        return False

    fdr_total = max(fBL, fDn, fH, fV)
    if fdr_total > conf.fdr:
        return False

    return True


def coalesce_pixels_to_centroid(
    feats: List[Feature2D],
    conf: HiCCUPSConfig,
) -> List[Feature2D]:
    """
    centroid 合并：
      - 在 (start1,start2) 平面上，用 cluster_radius_bp 做半径的聚类。
      - 每次以 observed 最大的像素为种子，吸收邻域内的所有像素，更新质心。
    """
    if not feats:
        return []

    uniq = {}
    for f in feats:
        key = (f.chr1, f.start1, f.start2)
        if key not in uniq or f.get_float(OBSERVED) > uniq[key].get_float(OBSERVED):
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
    chr_name: str,
    C_raw: bh.band_hic_matrix,
    kr: np.ndarray,
    expected_dist: np.ndarray,
    conf: HiCCUPSConfig,
):
    """
    单分辨率 HiCCUPS：
      - C_raw: raw Hi-C band matrix (band_hic_matrix)
      - kr: KR vector (len=N)
      - expected_dist: distance expected (len>=N)
    返回：
      - merged_loops: List[Feature2D]
      - peak matrix: np.ndarray (N×N)
    """
    N = C_raw.bin_num
    diag_num = C_raw.diag_num

    # 带状归一化：C_norm[i,k] = C_raw[i,k] * kr[i] * kr[i+k]
    norm_data = np.zeros_like(C_raw.data, dtype=np.float32)
    idx_rows = np.arange(N)
    for k in range(diag_num):
        j_idx = idx_rows + k
        valid = j_idx < N
        norm_data[valid, k] = (
            C_raw.data[valid, k] * kr[valid] * kr[j_idx[valid]]
        )

    C_norm = bh.band_hic_matrix(
        norm_data,
        diag_num=diag_num,
        mask=C_raw.mask,
        mask_row_col=C_raw.mask_row_col,
        band_data_input=True,
    )

    (
        observed_arr, eBL_arr, eDonut_arr, eH_arr, eV_arr,
        binBL_arr, binDonut_arr, binH_arr, binV_arr,
        histBL, histDonut, histH, histV,
        valid_mask,
        row_col_mask,
    ) = compute_evalues_and_hist(C_norm, expected_dist, conf, kr)

    observed = bh.band_hic_matrix(
        observed_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        band_data_input=True,
    )
    eBL = bh.band_hic_matrix(
        eBL_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        band_data_input=True,
    )
    eDonut = bh.band_hic_matrix(
        eDonut_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        band_data_input=True,
    )
    eH = bh.band_hic_matrix(
        eH_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        band_data_input=True,
    )
    eV = bh.band_hic_matrix(
        eV_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        band_data_input=True,
    )
    binBL = bh.band_hic_matrix(
        binBL_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        dtype=int,
        band_data_input=True,
    )
    binDonut = bh.band_hic_matrix(
        binDonut_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        dtype=int,
        band_data_input=True,
    )
    binH = bh.band_hic_matrix(
        binH_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        dtype=int,
        band_data_input=True,
    )
    binV = bh.band_hic_matrix(
        binV_arr,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=row_col_mask,
        dtype=int,
        band_data_input=True,
    )

    thrBL, fdrBL = compute_thresholds_and_fdr(histBL, conf)
    thrDo, fdrDo = compute_thresholds_and_fdr(histDonut, conf)
    thrH, fdrH = compute_thresholds_and_fdr(histH, conf)
    thrV, fdrV = compute_thresholds_and_fdr(histV, conf)

    # 第一遍：计算 peak 矩阵
    peak_data = np.zeros((N, diag_num), dtype=np.float32)
    for i, j, k in _iter_band_indices(observed):
        if not valid_mask[i, k]:
            continue
        o = observed.data[i, k]
        bBL = int(binBL.data[i, k])
        bDo = int(binDonut.data[i, k])
        bH = int(binH.data[i, k])
        bV = int(binV.data[i, k])
        sb = max(
            thrBL[bBL],
            thrDo[bDo],
            thrH[bH],
            thrV[bV],
        )
        peak_data[i, k] = o - sb

    peak = bh.band_hic_matrix(
        peak_data,
        diag_num=diag_num,
        mask=C_norm.mask,
        mask_row_col=C_norm.mask_row_col,
        band_data_input=True,
    )

    # 第二遍：local maxima + FDR + OE 阈值筛选
    candidates: List[Feature2D] = []

    for i, j, k in _iter_band_indices(peak):
        if not valid_mask[i, k]:
            continue

        diagDist = k
        if diagDist * conf.resolution > conf.max_loop_dist_bp:
            continue
        if diagDist <= conf.peak_width:
            continue

        val = peak.data[i, k]
        if val <= 0:
            continue

        # local maxima 检查（仅在带状区域内比较）
        pw = conf.peak_width
        max_val = -np.inf
        for ii in range(max(0, i - pw), min(N, i + pw + 1)):
            for jj in range(max(0, j - pw), min(N, j + pw + 1)):
                vv = peak[ii, jj]
                if ma.is_masked(vv):
                    continue
                if vv > max_val:
                    max_val = vv
        if val < max_val:
            continue

        e_bl = eBL.data[i, k]
        e_dn = eDonut.data[i, k]
        e_hh = eH.data[i, k]
        e_vv = eV.data[i, k]
        if min(e_bl, e_dn, e_hh, e_vv) <= 1e-6:
            continue

        o = observed.data[i, k]
        bBL = int(binBL.data[i, k])
        bDo = int(binDonut.data[i, k])
        bH = int(binH.data[i, k])
        bV = int(binV.data[i, k])

        f = generate_peak_feature(
            chr_name, conf.resolution, i, j,
            o, val,
            e_bl, e_dn, e_hh, e_vv,
            bBL, bDo, bH, bV,
        )
        add_fdr_to_feature(f, fdrBL, fdrDo, fdrH, fdrV)

        if fdr_thresholds_satisfied(f, conf):
            candidates.append(f)

    merged_loops = coalesce_pixels_to_centroid(candidates, conf)
    return merged_loops, peak


# =========================================================
# Step 5: multi-resolution HiCCUPS + merging
# =========================================================

def euclid_bp(f1: Feature2D, f2: Feature2D) -> float:
    dx = f1.start1 - f2.start1
    dy = f1.start2 - f2.start2
    return math.hypot(dx, dy)


def extract_reproducible_centroids(
    list_a: List[Feature2D],
    list_b: List[Feature2D],
    radius_bp: int
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
    peaks: List[Feature2D],
    centroids: List[Feature2D],
    radius_bp: int
) -> List[Feature2D]:
    out = []
    for f in peaks:
        for c in centroids:
            if f.chr1 == c.chr1 and euclid_bp(f, c) <= radius_bp:
                out.append(f)
                break
    return out


def extract_peaks_not_near_centroids(
    peaks: List[Feature2D],
    centroids: List[Feature2D],
    radius_bp: int
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
    peaks: List[Feature2D],
    max_dist_bp: int
) -> List[Feature2D]:
    out = []
    for f in peaks:
        if abs(f.start2 - f.start1) <= max_dist_bp:
            out.append(f)
    return out


def get_strong_peaks(
    peaks: List[Feature2D],
    min_observed: float
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
    仿 Juicer 的 HiCCUPSUtils.mergeAllResolutions:

    - 若存在 5kb 或 10kb：
        * 若两者都有：
            - 合并 5k & 10k（可重复 centroid 区域）
            - 补充 10k 仅存在的 peaks
            - 补充 5k 中近对角线 + 强 peaks
        * 若只有 5kb 或只有 10kb：直接用该分辨率列表
    - 若存在 25kb：
        * 若前面已存在 merged：
            - 提取 25k 中远离 merged centroid 的 peaks 加入
        * 否则 merged = 25k
    - 若 5/10/25 都没有：
        * 简单地 union 所有分辨率并去重
    """
    merged: List[Feature2D] = []
    list_altered = False

    has5 = 5000 in looplists and len(looplists[5000]) > 0
    has10 = 10000 in looplists and len(looplists[10000]) > 0
    has25 = 25000 in looplists and len(looplists[25000]) > 0

    # 处理 5k / 10k
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

    # 处理 25k
    if has25:
        twenty5 = looplists[25000]
        if list_altered:
            c25 = extract_reproducible_centroids(merged, twenty5, 2 * 25000)
            distant25 = extract_peaks_not_near_centroids(twenty5, c25, 2 * 25000)
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


def run_hiccups_multiresolution(
    chr_name: str,
    mats_by_res: Dict[int, np.ndarray],
    kr_by_res: Dict[int, np.ndarray],
    expected_by_res: Dict[int, np.ndarray],
    configs: Dict[int, HiCCUPSConfig],
):
    """
    多分辨率 HiCCUPS：
      - 对每个 resolution 跑 run_hiccups_single_resolution
      - 用 merge_all_resolutions 合并
    """
    looplists: Dict[int, List[Feature2D]] = {}
    for res, C_raw in mats_by_res.items():
        conf = configs[res]
        loops, _ = run_hiccups_single_resolution(
            chr_name,
            C_raw,
            kr_by_res[res],
            expected_by_res[res],
            conf,
        )
        looplists[res] = loops

    merged = merge_all_resolutions(looplists)
    return merged


def hiccups(
    chr_names: List[str],
    file_path: str,
    configs_by_res: Dict[int, HiCCUPSConfig],
    bedpe_path: str | None = None,
) -> List[Feature2D]:
    """
    对一个 chr，在多个分辨率上跑 HiCCUPS 并合并，输出最终 loops。
    """
    loops = run_hiccups_multiresolution(
        chr_name,
        mats_by_res,
        kr_by_res,
        expected_by_res,
        configs_by_res,
    )
    if bedpe_path is not None:
        write_bedpe(loops, bedpe_path)
    return loops


# =========================================================
# Step 6: BEDPE 输出
# =========================================================

def write_bedpe(loops: List[Feature2D], path: str):
    """
    输出 bedpe：
      chr1 start1 end1 chr2 start2 end2 score . . attrs
    其中 attrs 包含 O/E 和 FDR 信息等。
    """
    with open(path, "w") as f:
        for feat in loops:
            score = feat.get_float(PEAK)
            attrs_list = [
                f"O={feat.get_float(OBSERVED):.3g}",
                f"Ebl={feat.get_float(EXPECTEDBL):.3g}",
                f"Edonut={feat.get_float(EXPECTEDDONUT):.3g}",
                f"Eh={feat.get_float(EXPECTEDH):.3g}",
                f"Ev={feat.get_float(EXPECTEDV):.3g}",
            ]
            if FDRBL in feat.attrs:
                attrs_list.extend([
                    f"FDRbl={feat.get_float(FDRBL):.3g}",
                    f"FDRdonut={feat.get_float(FDRDONUT):.3g}",
                    f"FDRh={feat.get_float(FDRH):.3g}",
                    f"FDRv={feat.get_float(FDRV):.3g}",
                ])
            if NUMCOLLAPSED in feat.attrs:
                attrs_list.append(f"nPix={feat.get_float(NUMCOLLAPSED):.0f}")
            if RADIUS in feat.attrs:
                attrs_list.append(f"radius={feat.get_float(RADIUS):.1f}")

            attrs_str = ";".join(attrs_list)

            line = "\t".join([
                feat.chr1,
                str(feat.start1),
                str(feat.end1),
                feat.chr2,
                str(feat.start2),
                str(feat.end2),
                f"{score:.4g}",
                ".",
                ".",
                attrs_str,
            ])
            f.write(line + "\n")
