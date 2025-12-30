# -*- coding: utf-8 -*-
# HiCKRy.py

"""
HiCKRy: A tool for Hi-C contact matrix bias correction
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

This module provides functions to load Hi-C interaction data, compute bias values, and apply the Knight-Ruiz normalization algorithm.
It is designed to work with Hi-C data in the `.hic` format and can handle large datasets efficiently.
This code is modified from the original HiCKRy implementation by the Ay Lab, we added some features and optimizations for usability.

These are two ways to run this script:
1.  Command Line Interface (CLI):
    python HiCKRy.py -i interactions.gz -f fragments.gz -o bias_output.gz
2.  Import as a module:
    from bandhic.utils.HiCKRy import hickry
    bias, is_valid = hickry(hic_coo, verbose=True)
This module is part of the BandHiC package, which provides a banded Hi-C matrix toolkit.

Reference:
1. Kaul, A., Bhattacharyya, S. & Ay, F. Identifying statistically significant chromatin contacts from Hi-C data with FitHiC2. Nature Protocols 15, 991–1012 (2020).
"""

import gzip
import argparse
import sys
import numpy as np
import scipy.sparse as sps
import time

__all__ = [
    "compute_bin_bias",
]


def compute_bin_bias(
    hic_coo, verbose=False, bias_lowerbound=0.5, bias_upperbound=2
):
    """
    Compute bias values for Hi-C contact matrices using the Knight-Ruiz normalization algorithm.

    Parameters
    ----------
    hic_coo : scipy.sparse.coo_array
        A sparse COO matrix representing Hi-C contact data.
    verbose : bool, optional
        If True, print detailed information during processing. Default is False.

    Returns
    -------
    bias : numpy.ndarray
        A 1D array containing the bias values for each bin in the Hi-C matrix.
    is_valid : bool
        A boolean indicating whether the bias vector is valid (mean and median within typical range).

    Notes
    -----
    This function removes a specified percentage of the most sparse bins from the Hi-C matrix before computing
    the bias values. The Knight-Ruiz normalization algorithm is applied to the modified matrix to compute the bias.
    The function iteratively removes bins with low interaction counts until a valid bias vector is obtained.
    The bias vector is expected to have a mean and median close to 1, indicating balanced interaction frequencies across bins.
    """
    zero_ratio = (
        1
        - np.unique(hic_coo.row[hic_coo.row != hic_coo.col]).shape[0]
        / hic_coo.shape[0]
    )
    is_valid = False
    hic_coo = hic_coo + hic_coo.T
    for sparseToRemoveT in np.arange(zero_ratio, 1, 0.05):
        if verbose:
            print("%.2f of bins are removed" % sparseToRemoveT)
        try:
            bias, is_valid = returnBias(
                hic_coo,
                sparseToRemoveT,
                verbose,
                bias_lowerbound,
                bias_upperbound,
            )
        except Exception as e:
            if verbose:
                print(e)
            continue
        # checkBias(bias)
        if is_valid:
            return bias.ravel(), is_valid
    else:
        if not is_valid:
            print(
                "WARNING... Bias vector has a median or mean outside of typical range ({}, {}).".format(
                    bias_lowerbound, bias_upperbound
                )
            )
        return bias.ravel(), is_valid


def parse_args(arguments):
    parser = argparse.ArgumentParser(description="Check help flag")
    parser.add_argument(
        "-i",
        "--interactions",
        help="Path to the interactions file to generate bias values",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-f",
        "--fragments",
        help="Path to the interactions file to generate bias values",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Full path to output the generated bias file to",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-x",
        "--percentOfSparseToRemove",
        help="Percent of diagonal to remove",
        required=False,
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def loadfastfithicInteractions(interactionsFile, fragsFile, verbose):
    if verbose:
        print("Creating sparse matrix...")
    startT = time.time()
    with gzip.open(fragsFile, "rt") as frag:
        ctr = 0
        fragDic = {}
        revFrag = []
        for lines in frag:
            line = lines.rstrip().split()
            chrom = line[0]
            mid = int(line[2])
            if chrom not in fragDic:
                fragDic[chrom] = {}
            fragDic[chrom][mid] = ctr
            revFrag.append((chrom, mid))
            ctr += 1
    x = []
    y = []
    z = []
    with gzip.open(interactionsFile, "rt") as ints:
        for lines in ints:
            line = lines.rstrip().split()
            chrom1 = line[0]
            mid1 = int(line[1])
            chrom2 = line[2]
            mid2 = int(line[3])
            z.append(float(line[4]))
            x.append(fragDic[chrom1][mid1])
            y.append(fragDic[chrom2][mid2])
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    sparseMatrix = sps.coo_array((z, (x, y)), shape=(ctr, ctr))
    endT = time.time()
    if verbose:
        print("Sparse matrix creation took %s seconds" % (endT - startT))
    return sparseMatrix, revFrag


def returnBias(rawMatrix, perc, verbose, bias_lowerbound, bias_upperbound):
    rawMatrix = rawMatrix.copy()
    R = rawMatrix.sum()
    mtxAndRemoved = removeZeroDiagonalCSR(rawMatrix, perc, verbose)
    if verbose:
        print("Sparse rows removed")
    initialSize = rawMatrix.shape
    if verbose:
        print(
            "Initial matrix size: %s rows and %s columns"
            % (initialSize[0], initialSize[1])
        )
    rawMatrix = mtxAndRemoved[0]
    removed = mtxAndRemoved[1]
    newSize = rawMatrix.shape
    if verbose:
        print(
            "New matrix size: %s rows and %s columns"
            % (newSize[0], newSize[1])
        )
        print("Normalizing with KR Algorithm")
    result = knightRuizAlg(rawMatrix, verbose=verbose)
    colVec = result[0]
    # x = sps.diags(colVec.flatten(), 0, format='csr')

    bias = computeBiasVector(colVec)
    is_valid = checkBias(bias, verbose, bias_lowerbound, bias_upperbound)
    biasWZeros = addZeroBiases(removed, bias)
    return biasWZeros, is_valid


def removeZeroDiagonalCSR(mtx, perc, verbose):
    iteration = 0
    toRemove = []
    ctr = 0
    rowSums = mtx.sum(axis=0)
    rowSums = list(
        np.array(rowSums).reshape(
            -1,
        )
    )
    rowSums = list(enumerate(rowSums))
    rowSums.sort(key=lambda tup: tup[1])
    size = len(rowSums)
    rem = int(np.ceil(perc * size))
    if verbose:
        print("Removing %s percent of most sparse bins" % (perc))
        print("... corresponds to %s total rows" % (rem))
    valToRemove = rowSums[rem][1]
    # print(valToRemove)
    if verbose:
        print(
            "... corresponds to all bins with less than or equal to %s total interactions"
            % valToRemove
        )
    for value in rowSums:
        if value[1] <= valToRemove:
            toRemove.append(value[0])
    list(set(toRemove))
    toRemove.sort()
    mtx = dropcols_coo(mtx, toRemove)
    for num in toRemove:
        if iteration != 0:
            num -= iteration
        removeRowCSR(mtx, num)
        iteration += 1
    return [mtx, toRemove]


def computeBiasVector(x):
    one = np.ones((x.shape[0], 1))
    x = one / x
    sums = np.sum(x)
    avg = (1.0 * sums) / x.shape[0]
    bias = np.divide(x, avg)
    return bias


def addZeroBiases(lst, vctr):
    for values in lst:
        vctr = np.insert(vctr, values, np.nan, axis=0)
    return vctr


def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)  # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()


def removeRowCSR(mat, i):
    if not isinstance(mat, (sps.csr_array, sps.csr_matrix)):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i] : -n] = mat.data[mat.indptr[i + 1] :]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i] : -n] = mat.indices[mat.indptr[i + 1] :]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1 :]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def knightRuizAlg(A, tol=1e-6, f1=False, verbose=False):
    n = A.shape[0]
    e = np.ones((n, 1), dtype=np.float64)
    res = []

    Delta = 3
    delta = 0.1
    x0 = np.copy(e)
    g = 0.9

    etamax = eta = 0.1
    stop_tol = tol * 0.5
    x = np.copy(x0)

    rt = tol**2.0
    v = x * (A.dot(x))
    rk = 1.0 - v
    rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
    rho_km2 = rho_km1
    rout = rold = rho_km1

    MVP = 0  # we'll count matrix vector products
    i = 0  # outer iteration count

    if f1:
        print("it        in. it      res\n"),

    while rout > rt:  # outer iteration
        i += 1

        if i > 30:
            if verbose:
                print("iteration times is overflow!")
            break

        k = 0
        y = np.copy(e)
        innertol = max(eta**2.0 * rout, rt)
        while rho_km1 > innertol:  # inner iteration by CG
            k += 1
            if k == 1:
                Z = rk / v
                p = np.copy(Z)
                # rho_km1 = np.dot(rk.T, Z)
                rho_km1 = (rk.transpose()).dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            if k > 10:
                if verbose:
                    print("inner iteation is overflow!")
                break

            # update search direction efficiently
            w = x * A.dot(x * p) + v * p
            # alpha = rho_km1 / np.dot(p.T, w)[0,0]
            alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
            ap = alpha * p
            # test distance to boundary of cone
            ynew = y + ap
            if np.amin(ynew) <= delta:
                if delta == 0:
                    break

                ind = np.where(ap < 0.0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            y = np.copy(ynew)
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            # rho_km1 = np.dot(rk.T, Z)[0,0]
            rho_km1 = ((rk.transpose()).dot(Z))[0, 0]
        x *= y
        v = x * (A.dot(x))
        rk = 1.0 - v
        # rho_km1 = np.dot(rk.T, rk)[0,0]
        rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
        rout = rho_km1
        MVP += k + 1
        # update inner iteration stopping criterion
        rat = rout / rold
        rold = rout
        res_norm = rout**0.5
        eta_o = eta
        eta = g * rat
        if g * eta_o**2.0 > 0.1:
            eta = max(eta, g * eta_o**2.0)
        eta = max(min(eta, etamax), stop_tol / res_norm)
        if f1:
            print("%03i %06i %03.3f %e %e \n" % (i, k, res_norm, rt, rout))
            res.append(res_norm)
    if f1:
        print("Matrix - vector products = %06i\n" % (MVP))

    return [x, i, k]


def checkBias(biasvec, verbose, bias_lowerbound, bias_upperbound):
    is_valid = False
    std = np.std(biasvec)
    mean = np.mean(biasvec)
    median = np.median(biasvec)
    if mean < bias_lowerbound or mean > bias_upperbound:
        if verbose:
            print(
                "WARNING... Bias vector has a mean outside of typical range ({}, {}).".format(
                    bias_lowerbound, bias_upperbound
                )
            )
            print("Consider running with a larger -x option if problems occur")
            print("Mean\t%s" % mean)
            print("Median\t%s" % median)
            print("Std. Dev.\t%s" % std)
        is_valid = False
    elif median < bias_lowerbound or median > bias_upperbound:
        if verbose:
            print(
                "WARNING... Bias vector has a median outside of typical range ({}, {}).".format(
                    bias_lowerbound, bias_upperbound
                )
            )
            print("Consider running with a larger -x option if problems occur")
            print("Mean\t%s" % mean)
            print("Median\t%s" % median)
            print("Std. Dev.\t%s" % std)
        is_valid = False
    elif (
        bias_lowerbound <= median <= bias_upperbound
        and bias_lowerbound <= mean <= bias_upperbound
    ):
        if verbose:
            print("Mean\t%s" % mean)
            print("Median\t%s" % median)
            print("Std. Dev.\t%s" % std)
        is_valid = True
    else:
        is_valid = False
    return is_valid


def outputBias(biasCol, revFrag, outputFilePath):
    bpath = outputFilePath
    with gzip.open(bpath, "wt") as biasFile:
        ctr = 0
        for values in np.nditer(biasCol):
            chrommidTup = revFrag[ctr]
            chrom = chrommidTup[0]
            mid = chrommidTup[1]
            biasFile.write("%s\t%s\t%s\n" % (chrom, mid, values))
            ctr += 1


def main():
    args = parse_args(sys.argv[3:])
    matrix, revFrag = loadfastfithicInteractions(
        args.interactions, args.fragments, verbose=True
    )

    bias, is_valid = compute_bin_bias(matrix, verbose=True)

    if is_valid:
        outputBias(bias, revFrag, args.output)
    else:
        print("Error!!!")


if __name__ == "__main__":
    main()
