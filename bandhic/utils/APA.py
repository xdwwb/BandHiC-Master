"""
Lightweight Python port of the Aggregate Peak Analysis (APA) utilities.

The goal is to preserve the logic of the Java implementation
(`juicebox.tools.utils.juicer.apa`) while leaning on common Python
scientific packages.  All matrices are handled as dense ``numpy.ndarray``
instances; no chunking or sparse storage is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import bandhic as bh

__all = [
    "apa",
]


def _minimum_positive(values: np.ndarray) -> float:
    """Mimics ``MatrixTools.minimumPositive``."""
    positives = values[values > 0]
    if positives.size == 0:
        return 0.0
    return float(positives.min())


def standard_normalization(matrix: np.ndarray) -> np.ndarray:
    """Divide the matrix by its mean (or 1.0 if the mean is < 1)."""
    mean_val = float(np.nanmean(matrix))
    scale = 1.0 / max(1.0, mean_val)
    return np.nan_to_num(matrix, nan=0.0) * scale


def center_normalization(matrix: np.ndarray) -> np.ndarray:
    """Divide the matrix by its center value; fall back to min positive or 1."""
    center = matrix.shape[0] // 2
    center_val = float(matrix[center, center])
    if center_val == 0:
        center_val = _minimum_positive(matrix)
        if center_val == 0:
            center_val = 1.0
    return np.nan_to_num(matrix, nan=0.0) / center_val


def peak_enhancement(matrix: np.ndarray) -> float:
    """Central pixel divided by the average of the remaining pixels."""
    matrix = np.nan_to_num(matrix, nan=0.0)
    center = matrix.shape[0] // 2
    center_val = float(matrix[center, center])
    remainder_sum = float(matrix.sum() - center_val)
    remainder_avg = remainder_sum / (matrix.size - 1)
    if remainder_avg == 0:
        return np.inf
    return center_val / remainder_avg


def _percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    """
    Port of StatPercentile.evaluate: percentile of ``value`` within
    ``sorted_values`` expressed on a 0-100 scale.
    """
    n = len(sorted_values)
    left = int(np.searchsorted(sorted_values, value, side="left"))
    if left == n:
        return 100.0

    if sorted_values[left] > value:
        return max(0.0, (left / n) * 100.0)

    right = int(np.searchsorted(sorted_values, value, side="right"))
    if right == n:
        return 100.0

    positions = np.arange(left, right, dtype=float) / n
    return float(positions.mean() * 100.0)


def rank_percentile(matrix: np.ndarray) -> np.ndarray:
    """
    Apply percentile ranking to each non-zero entry in ``matrix``.
    Zero entries remain zero to mimic the Java behavior.
    """
    flat_sorted = np.sort(matrix, axis=None)
    ranked = np.zeros_like(matrix, dtype=float)
    it = np.nditer(matrix, flags=["multi_index"])
    for val in it:
        if val == 0:
            ranked[it.multi_index] = 0.0
        else:
            ranked[it.multi_index] = _percentile_rank(flat_sorted, float(val))
    return ranked


def extract_localized_data(
    contact_map: np.ndarray, x: int, y: int, window: int
) -> np.ndarray:
    """
    Slice a centered (2*window+1) square around (x, y). Values that fall
    outside the matrix bounds are padded with zeros, matching the bounded
    extraction used in the Java code path.
    """
    # contact_map = np.asarray(contact_map, dtype=float)
    size = 2 * window + 1
    result = np.zeros((size, size), dtype=float)

    row_start = max(0, x - window)
    row_end = min(contact_map.shape[0], x + window + 1)
    col_start = max(0, y - window)
    col_end = min(contact_map.shape[1], y + window + 1)

    dest_row_start = row_start - (x - window)
    dest_col_start = col_start - (y - window)

    result[
        dest_row_start : dest_row_start + (row_end - row_start),
        dest_col_start : dest_col_start + (col_end - col_start),
    ] = contact_map[row_start:row_end, col_start:col_end]
    return np.nan_to_num(result, nan=0.0)


def filter_loops_by_size(
    loops: Iterable[Tuple[int, int]],
    min_peak_dist: float,
    max_peak_dist: float,
) -> List[Tuple[int, int]]:
    """
    Mirror of APAUtils.filterFeaturesBySize with loops expressed as
    (x_bin, y_bin) tuples in bin units.
    """
    filtered: List[Tuple[int, int]] = []
    for x, y in loops:
        dist = abs(x - y)
        if dist < min_peak_dist:
            continue
        if dist > max_peak_dist:
            continue
        filtered.append((x, y))
    return filtered


@dataclass
class APARegionStatistics:
    """
    Region-based APA metrics ported from the Java implementation.
    """

    matrix: np.ndarray
    region_width: int
    peak2mean: float
    peak2ul: float
    peak2ur: float
    peak2ll: float
    peak2lr: float
    zscore_ll: float
    mean_ur: float

    @classmethod
    def from_matrix(
        cls, matrix: np.ndarray, region_width: int
    ) -> "APARegionStatistics":
        matrix = np.asarray(matrix, dtype=float)
        max_dim = matrix.shape[0]
        center = max_dim // 2
        central_val = float(matrix[center, center])

        mean_val = float((matrix.sum() - central_val) / (matrix.size - 1))
        peak2mean = central_val / mean_val if mean_val != 0 else np.inf

        ul = matrix[:region_width, :region_width]
        ur = matrix[:region_width, max_dim - region_width : max_dim]
        ll = matrix[max_dim - region_width : max_dim, :region_width]
        lr = matrix[
            max_dim - region_width : max_dim, max_dim - region_width : max_dim
        ]

        avg_ul = float(np.mean(ul))
        avg_ur = float(np.mean(ur))
        avg_ll = float(np.mean(ll))
        avg_lr = float(np.mean(lr))

        peak2ul = central_val / avg_ul if avg_ul != 0 else np.inf
        peak2ur = central_val / avg_ur if avg_ur != 0 else np.inf
        peak2ll = central_val / avg_ll if avg_ll != 0 else np.inf
        peak2lr = central_val / avg_lr if avg_lr != 0 else np.inf

        std_ll = float(np.std(ll))
        zscore_ll = (central_val - avg_ll) / std_ll if std_ll != 0 else np.inf

        return cls(
            matrix=matrix,
            region_width=region_width,
            peak2mean=peak2mean,
            peak2ul=peak2ul,
            peak2ur=peak2ur,
            peak2ll=peak2ll,
            peak2lr=peak2lr,
            zscore_ll=zscore_ll,
            mean_ur=avg_ur,
        )

    @property
    def region_corner_values(self) -> Tuple[float, float, float, float]:
        return (self.peak2ul, self.peak2ur, self.peak2ll, self.peak2lr)


@dataclass
class APAResult:
    """Container for APA outputs."""

    apa: np.ndarray
    normed_apa: np.ndarray
    center_normed_apa: np.ndarray
    rank_apa: np.ndarray
    enhancement: List[float]
    peak_numbers: Tuple[int, int, int]
    stats: APARegionStatistics

    def plot(
        self,
        output_path: str,
        type: str = "normed",
        title: str = "APA",
        use_cell_plotting: bool = True,
    ) -> None:
        """
        Optional matplotlib plot replicating the color scaling used by the Java
        APAPlotter. Save figure to ``output_path``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        if type == "normed":
            data = np.array(self.normed_apa, copy=True, dtype=float)
        elif type == "center_normed":
            data = np.array(self.center_normed_apa, copy=True, dtype=float)
        elif type == "rank":
            data = np.array(self.rank_apa, copy=True, dtype=float)
        elif type == "apa":
            data = np.array(self.apa, copy=True, dtype=float)
        else:
            raise ValueError(f"Unknown APA plot type: {type}")
        stats = APARegionStatistics.from_matrix(data, self.stats.region_width)
        title_with_stats = f"{title}, P2LL = {stats.peak2ll:.3f}"

        cmap = LinearSegmentedColormap.from_list("apa", ["white", "red"])

        if use_cell_plotting:
            max_allowed = 5 * stats.mean_ur
            data = np.clip(data, 0, max_allowed)
            vmin, vmax = data.min(), data.max()
        else:
            vmin, vmax = data.min(), data.max()

        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.imshow(
            data,
            # origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title_with_stats)
        ax.set_xlabel("Bin")
        ax.set_ylabel("Bin")

        cb = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("Signal", rotation=-90, va="bottom")

        # Annotate corner boxes like the Java plotter
        rw = self.stats.region_width
        center = data.shape[0] // 2
        offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for value, (dr, dc) in zip(stats.region_corner_values, offsets):
            row = center + dr * ((data.shape[0] // 2) - rw // 2)
            col = center + dc * ((data.shape[1] // 2) - rw // 2)
            ax.text(
                col,
                row,
                f"{value:.3f}",
                color="black",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
                ),
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


class APADataStack:
    """
    Accumulates per-loop cutouts and normalizations.
    Mirrors the structure of ``APADataStack`` in Java but scoped to a single run.
    """

    def __init__(self, size: int, region_width: int):
        self.size = size
        self.region_width = region_width
        self.apa_matrix = np.zeros((size, size), dtype=float)
        self.normed_apa_matrix = np.zeros((size, size), dtype=float)
        self.center_normed_apa_matrix = np.zeros((size, size), dtype=float)
        self.rank_apa_matrix = np.zeros((size, size), dtype=float)
        self.enhancement: List[float] = []
        self.count = 0

    def add_data(self, data: np.ndarray) -> None:
        """Add a single LxL cutout to the stack."""
        data = np.nan_to_num(np.asarray(data, dtype=float), nan=0.0)
        self.apa_matrix += data
        self.normed_apa_matrix += standard_normalization(data)
        self.center_normed_apa_matrix += center_normalization(data)
        self.rank_apa_matrix += rank_percentile(data)
        self.enhancement.append(peak_enhancement(data))
        self.count += 1

    def __add__(self, other: APADataStack) -> APADataStack:
        """Combine two APADataStack instances."""
        if self.size != other.size or self.region_width != other.region_width:
            raise ValueError(
                "Cannot add APADataStack instances of different sizes."
            )

        combined = APADataStack(size=self.size, region_width=self.region_width)
        combined.apa_matrix = self.apa_matrix + other.apa_matrix
        combined.normed_apa_matrix = (
            self.normed_apa_matrix + other.normed_apa_matrix
        )
        combined.center_normed_apa_matrix = (
            self.center_normed_apa_matrix + other.center_normed_apa_matrix
        )
        combined.rank_apa_matrix = self.rank_apa_matrix + other.rank_apa_matrix
        combined.enhancement = self.enhancement + other.enhancement
        combined.count = self.count + other.count
        return combined

    def merge(
        list_of_stacks: List[APADataStack], init_stack: APADataStack = None
    ) -> APADataStack:
        """Merge a list of APADataStack instances into one."""
        if not list_of_stacks:
            raise ValueError("No APADataStack instances to merge.")

        if init_stack is not None:
            combined = init_stack
        else:
            combined = APADataStack(
                size=list_of_stacks[0].size,
                region_width=list_of_stacks[0].region_width,
            )
        for stack in list_of_stacks:
            combined += stack
        return combined

    def threshold_plots(self, value: float) -> None:
        """Cap APA matrix entries at ``value`` (debug parity with Java)."""
        np.clip(self.apa_matrix, None, value, out=self.apa_matrix)

    def finalize(
        self, peak_numbers: Optional[Tuple[int, int, int]] = None
    ) -> APAResult:
        if self.count == 0:
            raise ValueError("No data added to APA stack.")

        scale = 1.0 / self.count
        normed = self.normed_apa_matrix * scale
        center_normed = self.center_normed_apa_matrix * scale
        rank = self.rank_apa_matrix * scale
        apa = self.apa_matrix * scale

        stats = APARegionStatistics.from_matrix(apa, self.region_width)
        peak_numbers = peak_numbers or (self.count, self.count, self.count)

        return APAResult(
            apa=apa,
            normed_apa=normed,
            center_normed_apa=center_normed,
            rank_apa=rank,
            enhancement=list(self.enhancement),
            peak_numbers=peak_numbers,
            stats=stats,
        )


def _chunk_loops(loops, n_chunks):
    """
    Split loops into n_chunks approximately equal parts.
    """
    if n_chunks <= 1:
        return [list(loops)]
    chunk_size = (len(loops) + n_chunks - 1) // n_chunks
    return [
        loops[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_chunks)
        if loops[i * chunk_size : (i + 1) * chunk_size]
    ]


def _apa_loop_chunk_worker(contact_map, loop_chunk, window, region_width):
    """
    Worker for a chunk of loops: creates its own APADataStack and processes chunk.
    """
    stack = APADataStack(size=2 * window + 1, region_width=region_width)
    for x, y in loop_chunk:
        cutout = extract_localized_data(contact_map, x, y, window)
        stack.add_data(cutout)
    return stack


def apa(
    hic_path: str,
    resolution: int,
    loops_df: pd.DataFrame,
    window: int = 10,
    region_width: int = 6,
    min_peak_dist: float = 0.0,
    max_peak_dist: float = 8_000_000,
    njobs: Optional[int] = -1,
) -> APAResult:
    """
    Compute Aggregate Peak Analysis (APA) around a set of loop anchors.

    This is a lightweight Python port of the Juicer/juicebox APA workflow.
    For each loop (x, y), we extract a (2*window+1)×(2*window+1) cutout centered
    at (x, y), accumulate cutouts across loops, and report common APA
    normalizations and region-based summary statistics.

    Notes
    -----
    * Loops are processed per chromosome. For each chromosome, a contact map is
      loaded from ``hic_path`` at the requested ``resolution`` using
      ``bandhic.straw_chr`` (with ``normalization='KR'`` in the current
      implementation).
    * Loop distance filtering is applied in **bin units** after converting loop
      coordinates to bins. By default, ``max_peak_dist`` is interpreted as a
      genomic distance in bp and converted to bins via ``resolution``.

    Parameters
    ----------
    hic_path:
        Path to an input ``.hic`` file.
    resolution:
        Hi-C bin size in base pairs.
    loops_df:
        Loop list as a DataFrame (e.g., BEDPE). The current implementation
        expects at least the columns ``'#chr1'``, ``'chr2'``, ``'x1'``, and
        ``'y1'`` (coordinates in bp). Only intra-chromosomal loops
        (``#chr1 == chr2``) are used.
    window:
        Number of bins to include on each side of the loop center; the final
        cutout size is ``2*window+1``.
    region_width:
        Corner box size (in bins) used for APA region statistics.
    min_peak_dist:
        Minimum loop distance from the diagonal (in bins, after binning).
    max_peak_dist:
        Maximum loop distance from the diagonal (in bp; converted to bins as
        ``max_peak_dist // resolution`` for filtering and matrix loading).
    njobs:
        Number of parallel worker processes for loop cutout extraction.
        ``-1`` uses up to ``os.cpu_count()`` workers (capped by the number of
        loops).

    Returns
    -------
    APAResult
        Aggregated APA matrices (raw and normalized), enhancement scores, peak
        counts, and region-based summary statistics.
    """
    import os
    from joblib import Parallel, delayed

    stack_merged = APADataStack(size=2 * window + 1, region_width=region_width)
    for chrom, loops_chr in loops_df.groupby("#chr1"):
        contact_map = bh.straw_chr(
            hic_path,
            chrom,
            resolution=resolution,
            diag_num=max_peak_dist // resolution + 1,
            normalization="KR",
        )
        loops_chr = loops_chr[loops_chr["#chr1"] == loops_chr["chr2"]]
        loops = list(
            zip(loops_chr["x1"] // resolution, loops_chr["y1"] // resolution)
        )

        filtered_loops = filter_loops_by_size(
            loops, min_peak_dist=min_peak_dist, max_peak_dist=max_peak_dist
        )
        if njobs == -1:
            n_jobs = min(os.cpu_count() or 1, len(filtered_loops))
        else:
            n_jobs = min(njobs, len(filtered_loops))
        if n_jobs <= 1 or len(filtered_loops) == 0:
            for x, y in filtered_loops:
                cutout = extract_localized_data(contact_map, x, y, window)
                stack_merged.add_data(cutout)
        else:
            loop_chunks = _chunk_loops(filtered_loops, n_jobs)
            stacks = Parallel(n_jobs=n_jobs)(
                delayed(_apa_loop_chunk_worker)(
                    contact_map, chunk, window, region_width
                )
                for chunk in loop_chunks
            )
            stack_merged = APADataStack.merge(stacks, init_stack=stack_merged)

    unique_loops = len(set(filtered_loops))
    peak_numbers = (len(filtered_loops), unique_loops, len(loops))
    return stack_merged.finalize(peak_numbers=peak_numbers)


# =========================================================
# Simple test function for HiCCUPS
# =========================================================
def test_apa_basic():
    """
    Minimal smoke test for HiCCUPS:
      - Runs hiccups on a small chromosome with one resolution.
      - Does not validate biological correctness; only checks that
        the pipeline executes end‑to‑end without errors.
    """
    import bandhic as bh
    import numpy as np
    import pandas as pd

    hic_path = "/Users/wwb/workspace-local/call_loop/data/GSE63525_GM12878_insitu_primary_replicate_combined.hic"
    resolutions = [5000]
    # hic_matrix = hic_matrix.todense()
    loops = pd.read_csv(
        "../../test/hiccups_test_output_5000_chr1.bedpe", sep="\t", header=0
    )
    chroms = loops["#chr1"].unique()
    apa_result = apa(
        hic_path=hic_path,
        resolution=resolutions[0],
        loops_df=loops,
        max_peak_dist=8_000_000,
        window=10,
        region_width=6,
        njobs=-1,
    )
    apa_result.plot(
        "../../test/test_apa_output.png",
        type="normed",
        title="Test APA Result",
    )
    print("APA test completed successfully.")


if __name__ == "__main__":
    test_apa_basic()
