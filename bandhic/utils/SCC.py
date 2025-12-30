import numpy as np

__all__ = ["scc"]


def scc(
    A,
    B,
    max_k=None,
    min_n=10,
    method="pearson",
):
    """
    Stratified correlation coefficient (SCC) for Hi-C using BandHiC.

    Parameters
    ----------
    A, B : band_hic_matrix
        Two BandHiC matrices (same shape, same diag_num)
    max_k : int
        Max diagonal (distance bin) to consider
    min_n : int
        Minimum number of valid pixels per stratum
    method : {"pearson", "spearman"}

    Returns
    -------
    scc : float
    per_diag : dict {k: rho_k}

    Examples
    --------
    >>> from bandhic import band_hic_matrix
    >>> from bandhic import scc
    >>> A = band_hic_matrix.straw_chr("example1.hic", "chr1", 10000, diag_num=200)
    >>> B = band_hic_matrix.straw_chr("example2.hic", "chr1", 10000, diag_num=200)
    >>> scc, per_diag = scc(A, B)
    """

    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if A.diag_num != B.diag_num:
        raise ValueError("A and B must have the same diag_num")

    if max_k is None:
        max_k = A.diag_num - 1

    if max_k > A.diag_num - 1:
        raise ValueError("max_k must be less than diag_num")

    num = 0.0
    den = 0.0
    per_diag = {}

    for k in range(1, max_k + 1):
        diag_len = A.shape[0] - k
        x = A.data[:diag_len, k]
        y = B.data[:diag_len, k]

        # mask invalid
        valid = np.isfinite(x) & np.isfinite(y)
        if A.mask is not None:
            valid &= ~A.mask[:diag_len, k]
        if B.mask is not None:
            valid &= ~B.mask[:diag_len, k]

        if valid.sum() < min_n:
            continue

        xk = x[valid]
        yk = y[valid]

        if method == "pearson":
            rho = np.corrcoef(xk, yk)[0, 1]
            if not np.isfinite(rho):
                continue
        elif method == "spearman":
            from scipy.stats import rankdata

            rx = rankdata(xk)
            ry = rankdata(yk)
            rho = np.corrcoef(rx, ry)[0, 1]
            if not np.isfinite(rho):
                continue
        else:
            raise ValueError("Unknown method")

        w = valid.sum() - 1
        num += w * rho
        den += w
        per_diag[k] = rho

    return num / den if den > 0 else np.nan, per_diag
