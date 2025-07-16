# -*- coding: utf-8 -*-
# _test_utils.py

"""
_test_utils: Utility functions for testing the BandHiC package.
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

This module provides functions to compare two band_hic_matrix objects and assert their equality.
"""

from .bandhic import band_hic_matrix
import numpy as np

__all__ = [
    "matrix_equal",
    "assert_band_matrix_equal",
]

def matrix_equal(a: band_hic_matrix, b: band_hic_matrix) -> bool:
    """
    Check if two band_hic_matrix objects are equal.

    Parameters
    ----------
    a : band_hic_matrix
        First band_hic_matrix object.
    b : band_hic_matrix
        Second band_hic_matrix object.

    Returns
    -------
    bool
        True if the matrices are equal, False otherwise.
    """
    if isinstance(a, band_hic_matrix) and isinstance(b, band_hic_matrix):
        if a.shape != b.shape:
            return False
        elif a.diag_num != b.diag_num:
            return False
        elif a.dtype != b.dtype:
            return False
        elif a.default_value != b.default_value:
            return False
        elif a.mask is not None and b.mask is not None:
            if not np.array_equal(a.mask, b.mask):
                return False
        elif a.mask is not None or b.mask is not None:
            return False
        elif a.mask_row_col is not None and b.mask_row_col is not None:
            if not np.array_equal(a.mask_row_col, b.mask_row_col):
                return False
        elif a.mask_row_col is not None or b.mask_row_col is not None:
            return False
        elif not np.array_equal(a.data[a.mask], b.data[b.mask]):
            return False
        else:
            return True
    else:
        return False

def assert_band_matrix_equal(a: band_hic_matrix, b: band_hic_matrix) -> bool:
    """
    Assert that two band_hic_matrix objects are equal.

    Parameters
    ----------
    a : band_hic_matrix
        First band_hic_matrix object.
    b : band_hic_matrix
        Second band_hic_matrix object.

    Raises
    ------
    AssertionError
        If the two matrices are not equal.
    """
    if isinstance(a, band_hic_matrix) and isinstance(b, band_hic_matrix):
        if a.shape != b.shape:
            raise AssertionError(
                "Shapes do not match: {} vs {}".format(a.shape, b.shape)
            )
        elif a.diag_num != b.diag_num:
            raise AssertionError(
                "Diagonal numbers do not match: {} vs {}".format(
                    a.diag_num, b.diag_num
                )
            )
        elif a.dtype != b.dtype:
            raise AssertionError(
                "Data types do not match: {} vs {}".format(a.dtype, b.dtype)
            )
        elif a.default_value != b.default_value:
            raise AssertionError(
                "Default values do not match: {} vs {}".format(
                    a.default_value, b.default_value
                )
            )
        elif a.mask is not None and b.mask is not None:
            if not np.array_equal(a.mask, b.mask):
                raise AssertionError(
                    "Masks do not match: {} vs {}".format(a.mask, b.mask)
                )
        elif a.mask is not None or b.mask is not None:
            raise AssertionError(
                "One of the matrices has a mask while the other does not."
            )
        elif a.mask_row_col is not None and b.mask_row_col is not None:
            if not np.array_equal(a.mask_row_col, b.mask_row_col):
                raise AssertionError(
                    "Row/column masks do not match: {} vs {}".format(
                        a.mask_row_col, b.mask_row_col
                    )
                )
        elif a.mask_row_col is not None or b.mask_row_col is not None:
            raise AssertionError(
                "One of the matrices has a row/column mask while the other does not."
            )
        elif not np.array_equal(a.data, b.data):
            raise AssertionError(
                "Data in band_hic_matrix objects do not match, matrices: {} vs {}".format(
                    a, b
                )
            )
        else:
            return True
    else:
        raise AssertionError(
            "Both inputs must be band_hic_matrix objects, got types: {} vs {}".format(
                type(a), type(b)
            )
        )