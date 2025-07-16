# -*- coding: utf-8 -*-
# _create.py

"""
_create: Functions to create band_hic_matrix objects with various initial values.
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

This module provides functions to create `band_hic_matrix` objects filled with ones, zeros, identity matrices, or a specified value.
"""

from typing import Tuple, Union
import numpy as np
from .bandhic import band_hic_matrix

__all__ = [
    "ones",
    "zeros",
    "eye",
    "full",
    "zeros_like",
    "ones_like",
    "eye_like",
    "full_like",
]

def ones(
    shape: Tuple[int, int], diag_num: int = 1, dtype: type = np.float64
) -> band_hic_matrix:
    """
    Create a band_hic_matrix object filled with ones.

    Parameters
    ----------
    shape : tuple of int
        Matrix shape as (bins, bins).
    diag_num : int, optional
        Number of diagonals to consider. Default is 1.
    dtype : data-type, optional
        The data type of the matrix. Default is np.float64.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object with all entries filled with ones.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.ones((5, 5), diag_num=3)
    >>> print(mat)
    band_hic_matrix(shape=(5, 5), diag_num=3, dtype=<class 'numpy.float64'>)
    """
    data = np.ones((shape[0], diag_num), dtype=dtype)
    return band_hic_matrix(
        data,
        diag_num=diag_num,
        dtype=dtype,
        band_data_input=True,
        default_value=1,
    )


def zeros(
    shape: Tuple[int, int], diag_num: int = 1, dtype: type = np.float64
) -> band_hic_matrix:
    """
    Create a band_hic_matrix object filled with zeros.

    Parameters
    ----------
    shape : tuple of int
        Matrix shape as (bins, bins).
    diag_num : int, optional
        Number of diagonals to consider. Default is 1.
    dtype : data-type, optional
        The data type of the matrix. Default is np.float64.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object with all entries filled with zeros.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.zeros((5, 5), diag_num=3)
    >>> print(mat)
    band_hic_matrix(shape=(5, 5), diag_num=3, dtype=<class 'numpy.float64'>)
    """
    data = np.zeros((shape[0], diag_num), dtype=dtype)
    return band_hic_matrix(
        data,
        diag_num=diag_num,
        dtype=dtype,
        band_data_input=True,
        default_value=0,
    )


def eye(
    shape: Tuple[int, int], diag_num: int = 1, dtype: type = np.float64
) -> band_hic_matrix:
    """
    Create a band_hic_matrix object filled as an identity matrix.

    Parameters
    ----------
    shape : tuple of int
        Matrix shape as (bins, bins).
    diag_num : int, optional
        Number of diagonals to consider. Default is 1.
    dtype : data-type, optional
        The data type of the matrix. Default is np.float64.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object with ones on the main diagonal and zeros elsewhere.

    Examples
    --------
    >>> mat = eye((5, 5), diag_num=3)
    >>> print(mat)
    band_hic_matrix(shape=(5, 5), diag_num=3, dtype=<class 'numpy.float64'>)
    """
    data = np.zeros((shape[0], diag_num), dtype=dtype)
    data[:, 0] = 1
    return band_hic_matrix(
        data,
        diag_num=diag_num,
        dtype=dtype,
        band_data_input=True,
        default_value=0,
    )


def full(
    shape: Tuple[int, int],
    fill_value: Union[int, float],
    diag_num: int = 1,
    dtype: type = np.float64,
) -> band_hic_matrix:
    """
    Create a band_hic_matrix object filled with a specified value.

    Parameters
    ----------
    shape : tuple of int
        Matrix shape as (bins, bins).
    fill_value : scalar
        Value to fill the matrix with.
    diag_num : int, optional
        Number of diagonals to consider. Default is 1.
    dtype : data-type, optional
        The data type of the matrix. Default is np.float64.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object with all entries filled with `fill_value`.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.full((5, 5), fill_value=7, diag_num=3)
    >>> print(mat)
    band_hic_matrix(shape=(5, 5), diag_num=3, dtype=<class 'numpy.float64'>)
    """
    data = np.full((shape[0], diag_num), fill_value, dtype=dtype)
    return band_hic_matrix(
        data,
        diag_num=diag_num,
        dtype=dtype,
        band_data_input=True,
        default_value=fill_value,
    )


def zeros_like(other: band_hic_matrix, dtype=None) -> band_hic_matrix:
    """
    Create a band_hic_matrix object matching another matrix, filled with zeros.

    Parameters
    ----------
    other : band_hic_matrix
        Reference matrix.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object matching `other`, filled with zeros.

    Examples
    --------
    >>> mat_ref = zeros((4, 4), diag_num=2)
    >>> mat = zeros_like(mat_ref)
    >>> isinstance(mat, band_hic_matrix)
    True
    """
    if dtype is None:
        dtype = other.dtype
    return zeros(other.shape, diag_num=other.diag_num, dtype=dtype)


def ones_like(other: band_hic_matrix, dtype=None) -> band_hic_matrix:
    """
    Create a band_hic_matrix object matching another matrix, filled with ones.

    Parameters
    ----------
    other : band_hic_matrix
        Reference matrix.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object matching `other`, filled with ones.

    Examples
    --------
    >>> mat_ref = zeros((4, 4), diag_num=2)
    >>> mat = ones_like(mat_ref)
    >>> isinstance(mat, band_hic_matrix)
    True
    """
    if dtype is None:
        dtype = other.dtype
    return ones(other.shape, diag_num=other.diag_num, dtype=dtype)


def eye_like(other: band_hic_matrix, dtype=None) -> band_hic_matrix:
    """
    Create a band_hic_matrix object matching another matrix, filled as an identity matrix.

    Parameters
    ----------
    other : band_hic_matrix
        Reference matrix.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object matching `other`, filled as an identity matrix.

    Examples
    --------
    >>> mat_ref = zeros((4, 4), diag_num=2)
    >>> mat = eye_like(mat_ref)
    >>> isinstance(mat, band_hic_matrix)
    True
    """
    if dtype is None:
        dtype = other.dtype
    return eye(other.shape, diag_num=other.diag_num, dtype=dtype)


def full_like(
    other: band_hic_matrix, fill_value: Union[int, float], dtype=None
) -> band_hic_matrix:
    """
    Create a band_hic_matrix object matching another matrix, filled with a specified value.

    Parameters
    ----------
    other : band_hic_matrix
        Reference matrix.
    fill_value : scalar
        Value to fill the matrix.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object matching `other`, filled with `fill_value`.

    Examples
    --------
    >>> mat_ref = zeros((4, 4), diag_num=2)
    >>> mat = full_like(mat_ref, fill_value=9)
    >>> isinstance(mat, band_hic_matrix)
    True
    """
    if dtype is None:
        dtype = other.dtype
    return full(other.shape, fill_value, diag_num=other.diag_num, dtype=dtype)