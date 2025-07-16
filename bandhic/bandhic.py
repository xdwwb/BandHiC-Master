# -*- coding: utf-8 -*-
# bandhic.py

"""
bandhic: A memory-efficient Python package for managing and analyzing Hi-C data down to sub-kilobase resolution.
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

Provides memory-efficient storage and manipulation of Hi-C contact matrices
using a banded matrix representation.
"""

import copy
import sys
import warnings
from numbers import Number
from typing import (Any, Callable, Dict, Iterable, Iterator, Optional, Tuple,
                    Union, Sequence)

import numpy as np
import numpy.ma as ma
from scipy.sparse import coo_array, coo_matrix, csr_array
import collections.abc

# -------------------------------------------------------------------------------
# Registry and helper for NumPy top-level function dispatch in __array_function__
_ARRAY_FUNCTION_DISPATCH: Dict[Callable, str] = {
    np.sum: "sum",
    np.prod: "prod",
    np.min: "min",
    np.max: "max",
    np.mean: "mean",
    np.var: "var",
    np.std: "std",
    np.ptp: "ptp",
    np.all: "all",
    np.any: "any",
    np.clip: "clip",
}

def register_array_function(func: Callable, method_name: str) -> None:
    """
    Register a mapping from a NumPy function to a band_hic_matrix method.

    Parameters
    ----------
    func : Callable
        The NumPy top-level function (e.g., np.clip) to intercept.
    method_name : str
        Name of the band_hic_matrix method to call for that function.
    """
    _ARRAY_FUNCTION_DISPATCH[func] = method_name
# -------------------------------------------------------------------------------

class band_hic_matrix(np.lib.mixins.NDArrayOperatorsMixin):
    """
    Symmetric banded matrix stored in upper-triangular format.
    This storage format is motivated by high-resolution Hi-C data characteristics:
        1. Symmetry of contact maps.
        2. Interaction frequency concentrated near the diagonal; long-range contacts are sparse (mostly zero).
        3. Contact frequency decays sharply with genomic distance.
    By storing only the main and a fixed number of super-diagonals as columns of a band matrix
    (diagonal-major storage: diagonal k stored in column k), we drastically reduce memory usage
    while enabling random access to Hi-C contacts. Additionally, mask and mask_row_col arrays
    track invalid or masked contacts to support downstream analysis.

    Operations on this band_hic_matrix are as simple as on a numpy.ndarray; users can ignore these storage details.

    This class stores only the main diagonal and up to (diag_num - 1) super-diagonals,
    exploiting symmetry by mirroring values for lower-triangular access.

    Attributes
    ----------
    shape : tuple of int
        Shape of the original full Hi-C contact matrix (bin_num, bin_num), 
        regardless of internal band storage format.
    dtype : data-type
        Data type of the matrix elements, compatible with numpy dtypes.
    diag_num : int
        Number of diagonals stored.
    bin_num : int
        Number of bins (rows/columns) of the Hi-C matrix.
    data : ndarray
        Array of shape (`bin_num`, `diag_num`) storing banded Hi-C data.
    mask : ndarray of bool or None
        Mask for individual invalid entries. Stored as a boolean ndarray of shape (`bin_num`, `diag_num`) with the same shape as `data`.
    mask_row_col : ndarray of bool or None
        Mask for entire rows and corresponding columns, indicating invalid bins.
        Stored as a boolean ndarray of shape (`bin_num`,). For computational convenience,
        row/column masks are also applied to the `mask` array to track masked entries.
    default_value : scalar
        Default value for out-of-band entries. Entries out of the banded region and not stored in the data array will be set to this value.

    Examples
    --------
    >>> import bandhic as bh
    >>> import numpy as np
    >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
    >>> mat.shape
    (4, 4)
    """

    def __init__(
        self,
        contacts: Union[coo_array, coo_matrix, tuple, np.ndarray],
        diag_num: int = 1,
        mask_row_col: Optional[np.ndarray] = None,
        mask: Optional[Tuple[np.ndarray,np.ndarray]] = None,
        dtype: Optional[type] = None,
        default_value: Union[int, float] = 0,
        band_data_input: bool = False,
    ) -> None:
        
        """
        Initialize a band_hic_matrix instance.

        Parameters
        ----------
        contacts : {coo_array, coo_matrix, tuple, ndarray}
            Input Hi-C data in COO format, tuple (data, (row, col)), full square array, or banded stored ndarray.
            For non-symmetric full arrays, only the upper-triangular part is used and the matrix is symmetrized.
            Full square arrays are not recommended for large matrices due to memory constraints.
        diag_num : int, optional
            Number of diagonals to store. Must be >=1 and <= matrix dimension. Default is 1.
        mask_row_col : ndarray of bool or indices, optional
            Mask for invalid rows/columns. Can be specified as:
            - A boolean array of shape (bin_num,) indicating which rows/columns to mask.
            - A list of indices to mask.
            Defaults to None (no masking).
        mask : ndarray pair of (row_indices, col_indices), optional
            Mask for invalid matrix entries. Can be specified as:
            - A tuple of two ndarray (row_indices, col_indices) listing positions to mask.
            Defaults to None (no masking).
        dtype : data-type, optional
            Desired numpy dtype; defaults to 'contacts' data dtype; compatible with numpy dtypes.
        default_value : scalar, optional
            Default value for unstored out-of-band entries. Default is 0.
        band_data_input : bool, optional
            If True, contacts is treated as precomputed band storage. Default is False.

        Raises
        ------
        ValueError
            If contacts type is invalid, diag_num out of range, or array shape invalid.

        Examples
        --------
        Initialize from a SciPy COO matrix:
        >>> import bandhic as bh
        >>> import numpy as np
        >>> from scipy.sparse import coo_matrix
        >>> coo = coo_matrix(([1, 2, 3], ([0, 1, 2],[0, 1, 2])), shape=(3,3))
        >>> mat1 = bh.band_hic_matrix(coo, diag_num=2)
        >>> mat1.data.shape
        (3, 2)

        Initialize from a tuple (data, (row, col)):
        >>> mat2 = bh.band_hic_matrix(([4, 5, 6], ([0, 1, 2],[2, 1, 0])), diag_num=1)
        >>> mat2.data.shape
        (3, 1)

        Initialize from a full dense array, only upper-triangular part is stored, lower part is symmetrized:
        >>> arr = np.arange(16).reshape(4,4)
        >>> mat3 = bh.band_hic_matrix(arr, diag_num=3)
        >>> mat3.data.shape
        (4, 3)

        Initialize with row/column mask, this masks entire rows and corresponding columns:
        >>> mask = np.array([True, False, False, True])
        >>> mat4 = bh.band_hic_matrix(arr, diag_num=2, mask_row_col=mask)
        >>> mat4.mask_row_col
        array([ True, False, False,  True])
        
        `mask_row_col` is also supported as a list of indices:
        >>> mat4 = bh.band_hic_matrix(arr, diag_num=2, mask_row_col=[0, 3])
        >>> mat4.mask_row_col
        array([ True, False, False,  True])
        
        Initialize from precomputed banded storage:
        >>> band = mat3.data.copy()
        >>> mat5 = bh.band_hic_matrix(band, band_data_input=True)
        >>> mat5.data.shape
        (4, 3)
        """
        
        self.default_value = default_value
        self.mask: Optional[np.ndarray] = None
        self.mask_row_col: Optional[np.ndarray] = None
        self.diag_num = diag_num

        if diag_num < 1:
            raise ValueError("diag_num must be greater than 0")
        if not isinstance(self.default_value, (Number, np.integer, np.floating)):
            raise ValueError(
                "default_value must be a number, {} type is provided.".format(
                    self.default_value.__class__
                )
            )

        if isinstance(contacts, (coo_array, coo_matrix)):
            self.bin_num = np.max(contacts.shape)
            if diag_num > self.bin_num:
                diag_num = self.bin_num  # Adjust diag_num to fit the matrix size
                warnings.warn(
                    "diag_num ({}) exceeds bin_num ({}) and has been adjusted to bin_num.".format(
                        diag_num, self.bin_num
                    ),
                    UserWarning,
                )
                # raise ValueError(
                #     "diag_num ({}) must be >=1 and <=bin_num ({})".format(
                #         diag_num, self.bin_num
                #     )
                # )
            row_idx = contacts.row
            col_idx = contacts.col
            data = contacts.data
            if not dtype:
                self.dtype = (
                    data.dtype
                )  # Default data type is float64 if not provided
            else:
                self.dtype = dtype
                data = data.astype(self.dtype)
        elif isinstance(contacts, tuple):
            data, (row_idx, col_idx) = contacts
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if not isinstance(row_idx, np.ndarray):
                row_idx = np.array(row_idx)
            if not isinstance(col_idx, np.ndarray):
                col_idx = np.array(col_idx)
            # Coordinates are zero-based, so add 1
            self.bin_num = max(np.max(row_idx), np.max(col_idx)) + 1
            if diag_num > self.bin_num:
                diag_num = self.bin_num  # Adjust diag_num to fit the matrix size
                warnings.warn(
                    "diag_num ({}) exceeds bin_num ({}) and has been adjusted to bin_num.".format(
                        diag_num, self.bin_num
                    ),
                    UserWarning,
                )
                # raise ValueError(
                #     "diag_num ({}) must be >=1 and <=bin_num ({})".format(
                #         diag_num, self.bin_num
                #     )
                # )
            if not dtype:
                self.dtype = data.dtype
            else:
                self.dtype = dtype
                data = data.astype(self.dtype)
        elif isinstance(contacts, np.ndarray):
            if band_data_input:
                # If band_data_format is True, treat contacts as band data
                self._init_from_band_data(
                    contacts, mask_row_col=mask_row_col, mask=mask, dtype=dtype
                )
                return
            else:
                self.bin_num = contacts.shape[0]
                if diag_num > self.bin_num:
                    diag_num = self.bin_num  # Adjust diag_num to fit the matrix size
                    warnings.warn(
                        "diag_num ({}) exceeds bin_num ({}) and has been adjusted to bin_num.".format(
                            diag_num, self.bin_num
                        ),
                        UserWarning,
                    )
                    # raise ValueError(
                    #     "diag_num ({}) must be >=1 and <=bin_num ({})".format(
                    #         diag_num, self.bin_num
                    #     )
                    # )
                if contacts.ndim != 2:
                    raise ValueError("contacts must be 2D array")
                if contacts.shape[0] != contacts.shape[1]:
                    raise ValueError("contacts must be square matrix")
                if not dtype:
                    self.dtype = contacts.dtype
                else:
                    self.dtype = dtype
                    contacts = contacts.astype(self.dtype)
        else:
            raise ValueError(
                "contacts must be coo_array, coo_matrix, (data,(row,col)) tuple or ndarray"
            )

        # Warn if default_value is not representable in dtype
        if self.dtype is not None:
            try:
                casted_value = np.array(self.default_value, dtype=self.dtype)
                if not np.isclose(
                    casted_value, self.default_value, rtol=1e-5, atol=1e-8
                ):
                    warnings.warn(
                        f"default_value ({self.default_value}) cannot be accurately represented in dtype {self.dtype}. "
                        f"Consider using a compatible dtype or adjusting the default_value.",
                        UserWarning,
                    )
            except Exception:
                warnings.warn(
                    f"default_value ({self.default_value}) is incompatible with dtype {self.dtype}.",
                    UserWarning,
                )

        self.shape = (self.bin_num, self.bin_num)
        self.diag_num = diag_num

        if isinstance(contacts, (coo_array, coo_matrix, tuple)):
            self.data = np.full(
                shape=(self.bin_num, self.diag_num),
                dtype=self.dtype,
                fill_value=default_value,
            )

            self.set_values(
                row_idx,
                col_idx,
                data,
            )  # Set initial values
        elif isinstance(contacts, np.ndarray):
            self.data = np.full(
                shape=(self.bin_num, self.diag_num),
                dtype=self.dtype,
                fill_value=default_value,
            )
            # Fill the data array with the provided contacts
            for i in range(self.bin_num):
                item_num = min(self.diag_num, self.bin_num - i)
                self.data[i, :item_num] = contacts[i, i : i + item_num]

        if mask is not None:
            self.add_mask(*mask)
        if mask_row_col is not None:
            self.add_mask_row_col(mask_row_col)

    def _init_from_band_data(
        self, band_data, mask_row_col=None, mask=None, dtype=None
    ):
        """
        Internal constructor from precomputed band storage.

        Parameters
        ----------
        band_data : ndarray
            2D array of shape (bin_num, diag_num) of band values.
        mask_row_col : ndarray of bool, optional
            Mask for invalid rows/columns.
        mask : ndarray of bool, optional
            Mask for invalid entries, with shape (bin_num, diag_num), turple (row_indices, col_indices) are not supported.
        dtype : data-type, optional
            Desired numpy dtype; defaults to band_data dtype; compatible with numpy dtypes.

        Raises
        ------
        ValueError
            If band_data is not 2D or `diag_num` > `bin_num`.

        Examples
        --------
        >>> import bandhic as bh
        >>> data = np.arange(6).reshape(3,2)
        >>> mat = bh.band_hic_matrix(data, band_data_input=True)
        >>> mat.shape
        (3, 3)
        """
        if band_data.ndim != 2:
            raise ValueError("band_data must be 2D array")
        self.bin_num = band_data.shape[0]
        self.shape = (self.bin_num, self.bin_num)
        self.diag_num = band_data.shape[1]  # Number of diagonals
        if self.diag_num > self.bin_num:
            raise ValueError(
                "diag_num ({}) must be >=1 and <=bin_num ({})".format(
                    self.diag_num, self.bin_num
                )
            )
        if not dtype:
            self.dtype = (
                band_data.dtype
            )  # Default data type is float64 if not provided
        else:
            self.dtype = dtype
        # Initialize data array
        if self.dtype != band_data.dtype:
            self.data = band_data.astype(self.dtype)
        else:
            self.data = (
                band_data  # Fill the data array with the provided band data
            )

        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise ValueError(
                    "mask must be a numpy array, {} type is provided.".format(
                        mask.__class__
                    )
                )
            if self.data.shape != mask.shape:
                raise ValueError("`mask` shape does not match `data` shape")
            elif mask.dtype != bool:
                raise ValueError("`mask` must be a boolean array")
            else:
                self.mask = mask
        if mask_row_col is not None:
            self.add_mask_row_col(mask_row_col)

    def _parsing_index(
        self, index: Tuple[Union[int, slice, Sequence[int]], Union[int, slice, Sequence[int]]]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[int], Optional[int]]:
        """
        Parse the provided index into row and column indices, handling slices and lists.

        Parameters:
            index (tuple): A tuple containing (row_idx, col_idx).

        Returns:
            tuple: Parsed row_idx, col_idx, number of rows, and number of columns.
        """
        row_idx_input, col_idx_input = index
        if isinstance(row_idx_input, (int, np.integer)):
            row_range = np.array([row_idx_input], dtype=int)
        elif isinstance(row_idx_input, list):
            row_range = np.array(row_idx_input, dtype=int)
        elif isinstance(row_idx_input, slice):
            start,stop,step = self._process_slice(row_idx_input)
            row_range = np.arange(
                start,
                stop,
                step,
                dtype=int,
            )
        elif isinstance(row_idx_input, np.ndarray):
            row_range = row_idx_input
        else:
            raise TypeError(
                "Invalid row index type. Must be int, list, ndarray, or slice, {} type are provided.".format(
                    row_idx_input.__class__
                )
            )

        if isinstance(col_idx_input, (int, np.integer)):
            col_range = np.array([col_idx_input], dtype=int)
        elif isinstance(col_idx_input, list):
            col_range = np.array(col_idx_input, dtype=int)
        elif isinstance(col_idx_input, slice):
            start, stop, step = self._process_slice(col_idx_input)
            col_range = np.arange(
                start,
                stop,
                step,
                dtype=int,
            )
        elif isinstance(col_idx_input, np.ndarray):
            col_range = col_idx_input
        else:
            raise TypeError(
                "Invalid column index type. Must be int, list, ndarray, or slice, {} type are provided.".format(
                    col_idx_input.__class__
                )
            )

        if isinstance(row_idx_input, slice) or isinstance(
            col_idx_input, slice
        ):
            # Create a meshgrid for row and column indices
            row_idx, col_idx = np.meshgrid(row_range, col_range)
            row_idx = row_idx.ravel()  # Flatten the row index array
            col_idx = col_idx.ravel()  # Flatten the column index array
            row_num = row_range.shape[0]
            col_num = col_range.shape[0]
            return row_idx, col_idx, row_num, col_num
        else:
            # Convert to array if not a slice or list
            row_idx = row_range
            col_idx = col_range
        return row_idx, col_idx, None, None

    # init mask array by mask invalid entries for band data
    def init_mask(self):
        """
        Initialize mask for invalid entries based on matrix shape.

        Raises
        ------
        ValueError
            If mask is already initialized.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.init_mask()
        """
        if self.mask is None:
            # Create a mask array with the same shape as the data
            self.mask = self._extract_raw_mask(self.data.shape)
        else:
            raise ValueError(
                "Mask already initialized. Use drop_mask() to clear the existing mask."
            )

    def _extract_raw_mask(self, shape: Tuple) -> np.ndarray:
        """
        Generate mask of invalid entries for banded storage.

        Parameters
        ----------
        shape : tuple of int
            (bin_num, diag_num) specifying the storage array shape.

        Returns
        -------
        ndarray of bool
            True at positions that does not store valid entries.
        """
        # mask = np.zeros(shape, dtype=bool)
        # # Fill the mask with True for invalid entries
        # bin_num = shape[0]
        # diag_num = shape[1]
        # for k in range(diag_num):
        #     row_start = bin_num - k
        #     mask[row_start:, k] = True
        rows = np.arange(shape[0])[:,None]
        cols = np.arange(shape[1])[None,:]
        mask = (rows + cols >= shape[0])
        return mask

    def add_mask(
        self,
        row_idx: Union[Sequence[int], np.ndarray],
        col_idx: Union[Sequence[int], np.ndarray],
    ) -> None:
        """
        Add mask entries for specified indices.

        Parameters
        ----------
        row_idx : array-like of int
            Row indices to mask.
        col_idx : array-like of int
            Column indices to mask.

        Raises
        ------
        ValueError
            If row_idx and col_idx have different shapes.

        Examples
        --------
        >>> import bandhic as bh
        >>> import numpy as np
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.add_mask([0, 1], [1, 2])
        """
        try:
            row_idx = np.array(row_idx, dtype=int)
            col_idx = np.array(col_idx, dtype=int)
        except Exception:
            raise TypeError("row_idx and col_idx must be array-like of integers")

        if row_idx.shape != col_idx.shape:
            raise ValueError("row_idx and col_idx must have the same shape")

        # Initialize mask array if not already present
        if self.mask is None:
            self.init_mask()

        if np.any(row_idx < 0) or np.any(row_idx >= self.bin_num) \
        or np.any(col_idx < 0) or np.any(col_idx >= self.bin_num):
            raise IndexError("row_idx and col_idx must be in [0, bin_num)")

        # Ensure row_idx <= col_idx by swapping invalid pairs
        row_idx, col_idx = self._swap_indices(row_idx, col_idx)

        # Identify indices within the stored band
        in_range = (col_idx - row_idx) < self.diag_num
        if np.any(~in_range):
            warnings.warn(
                "Some indices fall outside the stored band and will be ignored.",
                UserWarning,
            )

        # Compute diagonal offsets and set mask
        k_arr = col_idx[in_range] - row_idx[in_range]
        assert self.mask is not None
        self.mask[row_idx[in_range], k_arr] = True

    def _swap_indices(
        self, row_idx: np.ndarray, col_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure row_idx <= col_idx by swapping.

        Parameters
        ----------
        row_idx : array-like of int
        col_idx : array-like of int

        Returns
        -------
        tuple of ndarray
            Swapped (row_idx, col_idx).

        Warnings
        --------
        UserWarning
            If any col_idx < row_idx, indices are swapped.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> rows, cols = mat._swap_indices([1],[0])
        """
        col_idx = np.array(col_idx)
        row_idx = np.array(row_idx)
        mask_swap = col_idx < row_idx
        if np.any(mask_swap):
            warnings.warn(
                "Column indices should not be less than row indices; invalid indices are swapped.",
                UserWarning,
            )
            tmp = row_idx[mask_swap].copy()
            row_idx[mask_swap] = col_idx[mask_swap]
            col_idx[mask_swap] = tmp
        return row_idx, col_idx

    def add_mask_row_col(
        self,
        mask_row_col: Union[Sequence[int], Sequence[bool], np.ndarray, int]
    ) -> None:
        """
        Mask entire rows and corresponding columns.

        Parameters
        ----------
        mask_row_col : array-like of int or bool
            If boolean array of shape (bin_num,), True entries indicate rows/columns to mask.
            If integer array or sequence, treated as indices of rows/columns to mask.
        
        Raises
        ------
        ValueError
            If integer indices are out of bounds.
        Examples
        --------
        >>> import bandhic as bh
        >>> mask = np.array([True, False, False])
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> mat.add_mask_row_col(mask)
        """
        if isinstance(mask_row_col, list):
            mask_row_col = np.array(mask_row_col)
        elif isinstance(mask_row_col, (int, np.integer)):
            mask_row_col = np.array([mask_row_col])

        if isinstance(mask_row_col, np.ndarray):
            if mask_row_col.ndim != 1:
                raise ValueError(
                    "mask_row_col must be a 1D array, {} type is provided.".format(
                        mask_row_col.__class__
                    )
                )
            if (
                mask_row_col.shape[0] == self.bin_num
                and mask_row_col.dtype == bool
            ):
                self.mask_row_col = mask_row_col
            elif np.issubdtype(mask_row_col.dtype, np.integer):
                if np.any((mask_row_col >= self.bin_num) | (mask_row_col < 0)):
                    raise ValueError(
                        "row/colum indices must be in [0, bin_num]."
                    )
                self.mask_row_col = np.zeros(self.bin_num, dtype=bool)
                self.mask_row_col[mask_row_col] = True
            else:
                raise ValueError(
                    "mask_row_col must be a 1D array of bool or int, {} type is provided.".format(
                        mask_row_col.dtype
                    )
                )
        else:
            raise ValueError(
                "mask_row_col must be a 1D array or list, {} type is provided.".format(
                    mask_row_col.__class__
                )
            )

        if self.mask is None:
            self.mask = self._extract_raw_mask(self.data.shape)
        # Set invalid rows and columns in the mask using vectorized implementation
        self.mask[mask_row_col, :] = True
        cols = np.where(mask_row_col)[0]
        rows = np.arange(self.bin_num)
        grid_col, grid_row = np.meshgrid(cols, rows)
        diag_offset = grid_col - grid_row
        valid_mask = (diag_offset >= 0) & (diag_offset < self.diag_num)
        row_idx_array = grid_row[valid_mask]
        col_idx_array = diag_offset[valid_mask]
        self.mask[row_idx_array, col_idx_array] = True
        self.mask = np.logical_or(
            self.mask, self._extract_raw_mask(self.data.shape)
        )

    def unmask(self, indices: Optional[Tuple[np.ndarray, np.ndarray]]=None)->None:
        """
        Remove mask entries for specified indices or clear all.

        Parameters
        ----------
        indices : tuple of array-like or None
            Tuple (row_idx, col_idx) to unmask, or None to clear all.

        Raises
        ------
        Warning
            If no mask exists when trying to remove.

        Notes
        -----
        If `indices` is None, this will clear all masks (both entry-level and row/column), equivalent to `clear_mask()`.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> mat.unmask(( [0],[0] ))
        >>> mat.unmask()
        """
        if indices is None:
            # If no indices are provided, remove all masks
            self.clear_mask()
            return
        if self.mask is None:
            warnings.warn("No mask to remove.", UserWarning)
            return
        else:
            if isinstance(indices, (tuple)):
                row_idx, col_idx = indices
                row_idx = np.array(row_idx)  # Convert to array
                col_idx = np.array(col_idx)
            elif isinstance(indices, (np.ndarray)):
                row_idx = indices[:, 0]
                col_idx = indices[:, 1]
            else:
                raise ValueError(
                    "Invalid indices format. Must be tuple (row_idx, col_idx) or ndarray."
                )

            row_idx, col_idx = self._swap_indices(row_idx, col_idx)

            # Check for valid range
            in_range_idx = (col_idx - row_idx) < self.diag_num
            # Calculate diagonal offsets
            k_arr = col_idx[in_range_idx] - row_idx[in_range_idx]
            # Update mask for valid entries
            self.mask[row_idx[in_range_idx], k_arr] = False

    def _set_mask_by_array(self, mask_array):
        """
        Replace mask with provided boolean array.

        Parameters
        ----------
        mask_array : ndarray of bool
            New mask array.

        Examples
        --------
        >>> import bandhic as bh
        >>> new_mask = np.zeros((4,2), dtype=bool)
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat._set_mask_by_array(new_mask)
        """
        if not isinstance(mask_array, np.ndarray):
            raise ValueError(
                "mask_array must be a numpy array, {} type is provided.".format(
                    mask_array.__class__
                )
            )
        if not mask_array.dtype == bool:
            raise ValueError(
                "mask_array must be a boolean array, {} type is provided.".format(
                    mask_array.dtype
                )
            )
        if mask_array.ndim != 2:
            raise ValueError(
                "mask_array must be a 2D array, {} type is provided.".format(
                    mask_array.ndim
                )
            )
        if mask_array.shape != self.data.shape:
            raise ValueError(
                "mask_array shape does not match data shape {} != {}".format(
                    mask_array.shape, self.data.shape
                )
            )
        if self.mask is not None:
            warnings.warn(
                "Existing mask will be replaced with the new mask.",
                UserWarning,
            )
        self.mask = mask_array

    def get_mask(self):
        """
        Get current mask array.

        Returns
        -------
        ndarray or None
            Current mask array or None if no mask.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mask = mat.get_mask()
        """
        return self.mask

    def get_mask_row_col(self):
        """
        Get current row/column mask.

        Returns
        -------
        ndarray or None
            Current row/column mask or None if no mask.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mask_row_col = mat.get_mask_row_col()
        """
        return self.mask_row_col

    def clear_mask(self) -> None:
        """
        Clear all masks (both entry-level and row/column-level).

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.clear_mask()
        """
        self.mask = None
        self.mask_row_col = None

    def drop_mask(self) -> None:
        """
        Clear the current mask by entry-level,  but retain the row/column mask.
        
        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2, mask=(np.array([[0, 1], [1, 2]])))
        >>> mat.drop_mask()
        
        Notes
        -----
        This method is useful when you want to keep the row/column mask but clear the entry-level mask.
        It will not affect the row/column mask, allowing you to maintain the masking of entire rows and columns.
        >>> mat.mask
        """
        self.mask = None
        # If mask_row_col is set, we need to add the row/column mask to the mask array.
        if self.mask_row_col is not None:
            self.add_mask_row_col(self.mask_row_col)
        
    def drop_mask_row_col(self):
        """
        Clear the current row/column mask.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.drop_mask_row_col()
        """
        # Set invalid rows and columns in the mask using vectorized implementation
        if self.mask is not None and self.mask_row_col is not None:
            self.mask[self.mask_row_col, :] = False
            cols = np.where(self.mask_row_col)[0]
            rows = np.arange(self.bin_num)
            grid_col, grid_row = np.meshgrid(cols, rows)
            diag_offset = grid_col - grid_row
            valid_mask = (diag_offset >= 0) & (diag_offset < self.diag_num)
            row_idx_array = grid_row[valid_mask]
            col_idx_array = diag_offset[valid_mask]
            self.mask[row_idx_array, col_idx_array] = False
            row_mask = self._extract_raw_mask(self.data.shape)
            self.mask = np.logical_or(self.mask, row_mask)
        self.mask_row_col = None

    def count_masked(self):
        """
        Count the number of masked entries in the banded matrix.
        This counts the number of entries in the upper triangular part of the matrix,
        excluding the diagonal and valid entries.
        Returns
        -------
        int
            Number of valid entries in the banded matrix.

            Examples
            --------
            >>> import bandhic as bh
            >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
            >>> mat.count_masked()
            0
        """
        if self.mask is None:
            return 0
        elif self.mask_row_col is None:
            return (
                self.mask[~self._extract_raw_mask(self.data.shape)].sum() * 2
                - self.mask[:, 0].sum()
            )
        else:
            # Count masked entries in the upper triangular part
            mask = self.mask.copy()
            mask[self.mask_row_col, :] = False
            cols = np.where(self.mask_row_col)[0]
            rows = np.arange(self.bin_num)
            grid_col, grid_row = np.meshgrid(cols, rows)
            diag_offset = grid_col - grid_row
            valid_mask = (diag_offset >= 0) & (diag_offset < self.diag_num)
            row_idx_array = grid_row[valid_mask]
            col_idx_array = diag_offset[valid_mask]
            mask[row_idx_array, col_idx_array] = False
            raw_mask = self._extract_raw_mask(self.data.shape)
            mask = np.logical_or(mask, raw_mask)
            # Exclude the masked rows and columns
            # and count the remaining masked entries
            masked_entries = mask[~raw_mask].sum()
            masked_count = (
                masked_entries * 2
                - mask[:, 0].sum()
                + self.mask_row_col.sum() * self.bin_num * 2
                - self.mask_row_col.sum() ** 2
            )
            return masked_count

    def count_unmasked(self):
        """
        Count the number of unmasked entries in the banded matrix.

        Returns
        -------
        int
            Number of unmasked entries in the banded matrix.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.count_unmasked()
        16
        """
        if self.mask is None:
            return self.bin_num**2
        else:
            return self.bin_num**2 - self.count_masked()

    def count_in_band_masked(self):
        """
        Count the number of masked entries in the in-band region.

        Returns
        -------
        int
            Number of masked entries in the in-band region.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.count_in_band_masked()
        0
        """
        if self.mask is None:
            return 0
        result = (
            self.mask[~self._extract_raw_mask(self.data.shape)].sum() * 2
            - self.mask[:, 0].sum()
        )
        return result

    def count_out_band_masked(self):
        """
        Count the number of masked entries in the out-of-band region.

        Returns
        -------
        int
            Number of masked entries in the out-of-band region.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.count_out_band_masked()
        0
        """
        if self.mask is None:
            return 0
        result = self.count_masked() - self.count_in_band_masked()
        return result

    def count_in_band(self):
        """
        Count the number of valid entries in the in-band region.

        Returns
        -------
        int
            Number of valid entries in the in-band region.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.count_in_band()
        10
        """
        return (
            np.sum(np.arange(self.bin_num, self.bin_num - self.diag_num, -1))
            * 2
            - self.bin_num
        )

    def count_out_band(self):
        """
        Count the number of valid entries in the out-of-band region.

        Returns
        -------
        int
            Number of valid entries in the out-of-band region.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2)
        >>> mat.count_out_band()
        6
        """
        return self.bin_num**2 - self.count_in_band()

    # fill the masked values with default_value
    def filled(self, fill_value: Union[int,float,None]= None, copy: bool = True) -> "band_hic_matrix":
        """
        Fill masked entries in data with default value.

        Parameters
        ----------
        fill_value : scalar
            Value to assign to masked entries.

        Raises
        ------
        ValueError
            If no mask is initialized.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(4), diag_num=2, mask = ([0,1],[1,2]))
        >>> mat.filled()
            band_hic_matrix(shape=(4, 4), diag_num=2, dtype=float64)
        """
        if self.mask is None:
            warnings.warn("No masked location to fill.", UserWarning)
            return self
        else:
            if fill_value is None:
                fill_value = self.default_value
            if isinstance(fill_value, (int, float, np.integer, np.floating)):
                if copy:
                    result = self.copy()
                else:
                    result = self
                result.data[result.mask] = fill_value
                result.clear_mask()
                result.default_value = fill_value
                return result
            else:
                raise ValueError("default_values must be int or float")

    def _process_slice(self, slice_input: slice) -> Tuple[int, int, int]:
        """
        Process a slice input to extract start, stop, and step values.

        Parameters:
            slice_input (slice): A slice object.

        Returns:
            tuple: Start, stop, and step values.
        """
        start = slice_input.start
        stop = slice_input.stop
        step = slice_input.step
        if start is None:
            start = 0  # Default start is 0
        if stop is None:
            stop = self.shape[0]  # Default stop is the shape of the matrix
        if step is None:
            step = 1  # Default step is 1
        return start, stop, step

    def __getitem__(
        self,
        index: Union[Tuple[Union[int, slice, np.ndarray, list], Union[int, slice, np.ndarray, list]], 
                     slice, 
                     "band_hic_matrix"]
    ) -> Union[Number, np.ndarray, np.ma.MaskedArray, "band_hic_matrix"]:
        """
        Retrieve matrix entries or submatrix using NumPy-like indexing.

        Supports:
        - Integer indexing: mat[i, j] returns a single value.
        - Slice indexing: mat[i:j, i:j] returns a band_hic_matrix for square slices.
        - Single-axis slice: mat[i:j] returns a band_hic_matrix same as mat[i:j, i:j].
        - Fancy (array) indexing: mat[[i1, i2], [j1, j2]] returns an ndarray or MaskedArray according to `mask`.
        - Mixed indexing: combinations of integer, slice, and array-like indices.
        - Boolean indexing: 'band_hic_matrix' object with dtype `bool` can be used to index entries.

        When both row and column indices specify the same slice (or a single slice is provided), 
        a new band_hic_matrix representing that square submatrix is returned. New submatrix is the view of the original matrix,
        sharing the same data and mask. If the mask or data is altered in the submatrix,
        the original matrix will reflect those changes as well.
        If a single integer index is provided for both row and column, a scalar value is returned.
        If a mask is set, masked entries will return as `numpy.ma.masked` for scalars, or as a `numpy.ma.MaskedArray` for arrays.
        If a mask is not set, the scalar value is returned directly, or a numpy.ndarray for arrays.
        If a square slice is provided, a new band_hic_matrix is returned with the same diagonal number and shape as the original matrix.
        If a single slice is provided, it returns a band_hic_matrix with the same diagonal number and shape as the original matrix.
        If fancy indexing is used, it returns a numpy.ndarray or numpy.ma.MaskedArray depending on whether the mask is set.
        In all other cases, a numpy.ndarray (if no mask) or numpy.ma.MaskedArray (if mask present) 
        is returned.

        Parameters
        ----------
        index : int, slice, band_hic_matrix, array-like of int, or tuple of these
            Index expression for rows and columns. May be:
            - A pair `(row_idx, col_idx)` of ints, slices, or array-like for mixed indexing.
            - A single slice selecting a square region.
            - A `band_hic_matrix` object with dtype of `bool` for boolean indexing.

        Returns
        -------
        scalar or ndarray or MaskedArray or band_hic_matrix
            - scalar : when both row and column are integer indices.
            - numpy.ndarray : for fancy or mixed indexing without mask.
            - numpy.ma.MaskedArray : for fancy or mixed indexing when mask is set.
            - band_hic_matrix : for square slice results.

        Raises
        ------
        ValueError
            If a slice step is not 1, or if indices are out of bounds.

        Examples
        --------
        >>> import numpy as np
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(16).reshape(4,4), diag_num=2)

        # Single-element access (scalar)
        >>> mat[1, 2]
        6

        # Masked element returns masked
        >>> mat2 = bh.band_hic_matrix(np.eye(4), dtype=int, diag_num=2, mask=([0],[1]))
        >>> mat2[0, 1]
        masked

        # Square submatrix via two-slice indexing returns band_hic_matrix
        >>> sub = mat[1:3, 1:3]
        >>> isinstance(sub, bh.band_hic_matrix)
        True

        # Single-axis slice returns band_hic_matrix for square region
        >>> sub2 = mat[0:2]  # equivalent to mat[0:2, 0:2]
        >>> isinstance(sub2, bh.band_hic_matrix)
        True

        # Fancy indexing returns ndarray or MaskedArray
        >>> arr = mat[[0,2,3], [1,2,0]]
        >>> isinstance(arr, np.ndarray)
        True

        >>> mat.add_mask([0,1],[1,2])  # Add mask to some entries
        >>> masked_arr = mat[[0,1], [1,2]]
        >>> isinstance(masked_arr, np.ma.MaskedArray)
        True
        
        # Boolean indexing with band_hic_matrix
        >>> mat3 = bh.band_hic_matrix(np.eye(4), diag_num=2, mask=([0,1],[1,2]))
        >>> bool_mask = mat3 > 0  # Create a boolean mask
        >>> result = mat3[bool_mask]  # Use boolean mask for indexing
        >>> isinstance(result, np.ma.MaskedArray)
        True
        >>> result
        masked_array(data=[1.0, 1.0, 1.0, 1.0],
                    mask=[False, False, False, False],
            fill_value=0.0)
        """
        
        if (
            isinstance(index, tuple)
            and isinstance(index[0], slice)
            and isinstance(index[1], slice)
        ):
            # If both row and column indices are slices, return a submatrix
            if index[0] == index[1]:
                index = index[0]
        if isinstance(index, slice):
            start, stop, step = self._process_slice(
                index
            )  # Process slice input
            if step != 1:
                raise ValueError(
                    "The step of slice must be 1 for bandhic slicing"
                )

            # Adjust diagonal number
            slice_diag_num = min(self.diag_num, stop - start)
            slice_bin_num = stop - start  # Adjust bin number

            slice_data = self.data[
                start:stop, :slice_diag_num
            ]  # Slice the data
            slice_shape = (slice_bin_num, slice_bin_num)  # Adjust shape
            if self.mask is not None:
                slice_mask = self.mask[start:stop, :slice_diag_num]
                slice_mask = slice_mask & self._extract_raw_mask(
                    slice_mask.shape
                )
            else:
                slice_mask = None
            if self.mask_row_col is not None:
                slice_mask_row_col = self.mask_row_col[start:stop]
                if not np.any(slice_mask_row_col):
                    slice_mask_row_col = None
                    # If any row/col is masked, apply the mask
            else:
                slice_mask_row_col = None
            slice_dtype = self.dtype
            return self.__class__(
                slice_data,
                diag_num=slice_diag_num,
                mask=slice_mask,
                mask_row_col=slice_mask_row_col,
                dtype=slice_dtype,
                default_value=self.default_value,
                band_data_input=True,
            )  # Return sliced matrix
        elif isinstance(index, self.__class__):
            if index.shape != self.shape:
                raise ValueError(
                    "The shape of the input band_hic_matrix must match the original matrix shape."
                )
            if index.dtype != np.bool_:
                raise ValueError(
                    "The dtype of the input band_hic_matrix must be bool."
                )
            idx_bool = index.data.copy()  # Get the boolean mask
            idx_bool = np.logical_and(
                idx_bool, ~self._extract_raw_mask(self.data.shape)
            )
            result = self.data[idx_bool]  # Use the mask to index data
            if self.mask is not None or index.mask is not None:
                mask = np.logical_or(
                    index.mask, self.mask
                )
                result = ma.MaskedArray(
                    result,
                    mask=mask[idx_bool],
                    fill_value=self.default_value,
                )
            return result  # Return masked array)
        else:
            row_idx, col_idx, row_num, col_num = self._parsing_index(
                index
            )  # Parse the index
            if len(row_idx) == 1 and len(col_idx) == 1:
                # If both row and column are single indices, return a scalar
                    return self.get_values(row_idx, col_idx)[0]
            elif isinstance(index[0], slice) or isinstance(index[1], slice):
                # Reshape if using slices
                return self.get_values(row_idx, col_idx).reshape(
                    row_num, col_num
                )
            else:
                # Return flat values
                return self.get_values(row_idx, col_idx)

    def __setitem__(
        self,
        index: Union[Tuple[Union[int, slice, np.ndarray, list], Union[int, slice, np.ndarray, list]],
                     slice, 
                     "band_hic_matrix"],
        values: Union[Number, np.ndarray, list],
    ) -> None:
        """
        Assign values to matrix entries using NumPy-like indexing.
        
        Parameters
        ----------
        index : int, tuple of (row_idx, col_idx), slice, or band_hic_matrix
            Index expression for rows and columns. May be:
            - A single integer for both row and column.
            - A tuple of row and column indices (can be int, slice, or array-like).
            - A single slice selecting a square region.
            - A `band_hic_matrix` object with dtype of `bool` for boolean indexing.
        values : scalar or array-like
            Values to assign. Can be a single scalar or an array-like object.
        Raises
        ------
        ValueError
            If index is a slice with step not equal to 1, or if indices exceed matrix dimensions.
        TypeError
            If `values` is not a scalar or array-like object.

        Supports:
        - Integer indexing: mat[i, j] = value assigns to a single element.
        - Slice indexing: mat[i:j, i:j] = array or scalar assigns to a square submatrix.
        - Single-axis slice: mat[i:j] = ... is equivalent to mat[i:j, i:j].
        - Fancy (array) indexing: mat[[i1, i2], [j1, j2]] = array or scalar for scattered assignments.
        - Mixed indexing: combinations of integer, slice, and array-like indices.
        - Boolean indexing: boolean mask (another band_hic_matrix with dtype=bool) selects entries to set.

        Examples
        --------
        >>> import bandhic as bh
        >>> import numpy as np
        >>> mat = bh.band_hic_matrix(np.zeros((4,4)), diag_num=2, dtype=int)
        
        # Single element assignment
        >>> mat[1, 2] = 5
        >>> mat[1, 2]
        5
        
        # Slice assignment to square submatrix
        >>> mat[0:2, 0:2] = [[1, 2], [2, 4]]
        >>> mat[0:2, 0:2].todense()
        array([[1, 2],
               [2, 4]])
               
        # Single-axis slice assignment (equivalent square slice)
        >>> mat[2:4] = 0
        >>> mat[2:4].todense()
        array([[0, 0],
               [0, 0]])
        
        # Fancy indexing for scattered assignments
        >>> mat[[0, 3], [1, 2]] = [7, 8]
        >>> mat[0, 1], mat[3, 2]
        (7, 8)
        
        # Boolean mask assignment
        >>> mat2 = bh.band_hic_matrix(np.eye(4), diag_num=2, dtype=int)
        >>> bool_mask = mat2 > 0
        >>> mat2[bool_mask] = 9
        >>> mat2.todense()
        array([[9, 0, 0, 0],
              [0, 9, 0, 0],
              [0, 0, 9, 0],
              [0, 0, 0, 9]])

        Notes
        -----
        - Assigning to masked entries updates underlying data but does not automatically unmask.
        - For multidimensional assignments, scalar values broadcast to all selected positions, while array values must match the number of targeted elements.
        - If a boolean mask is used, it must be a `band_hic_matrix` with dtype `bool` and the same shape as the original matrix.
        - If a single slice is provided, it behaves like mat[i:j, i:j] for square submatrices.
        """
        # Handle single slice: mat[i:j] means mat[i:j, i:j]
        if isinstance(index, self.__class__):
            if index.shape != self.shape:
                raise ValueError(
                    "The shape of the input band_hic_matrix must match the original matrix shape."
                )
            if index.dtype != np.bool_:
                raise ValueError(
                    "The dtype of the input band_hic_matrix must be bool."
                )
            idx_bool = index.data.copy()
            idx_bool = np.logical_and(
                idx_bool, ~self._extract_raw_mask(self.data.shape)
            )
            if isinstance(values, (int, float, np.integer, np.floating)):
                self.data[idx_bool] = values
            elif isinstance(values, (np.ndarray, list)):
                if len(values) != np.sum(idx_bool):
                    raise ValueError(
                        "The length of values must match the number of True entries in the mask."
                    )
                self.data[idx_bool] = values
            return
        # Fallback: parse arbitrary index
        if isinstance(index, slice):
            index = (index, index)  # Convert to tuple for consistency
        row_idx, col_idx, row_num, col_num = self._parsing_index(
            index
        )  # Parse the index
        if row_num is None or col_num is None or isinstance(
            values, (int, float, np.integer, np.floating)
        ):
            self.set_values(row_idx, col_idx, values)  # Set the values
        else:
            values = np.array(values)
            self.set_values(row_idx, col_idx, values.ravel(order='F'))  # Set the values

    def get_values(
        self,
        row_idx: Union[np.ndarray, Iterable[int]],
        col_idx: Union[np.ndarray, Iterable[int]],
    ) -> Union[np.ndarray, ma.MaskedArray]:
        """
        Retrieve values considering mask.

        Parameters
        ----------
        row_idx : array-like of int
            Row indices.
        col_idx : array-like of int
            Column indices.

        Returns
        -------
        ndarray or MaskedArray
            Retrieved values; masked entries yield masked results.

        Raises
        ------
        ValueError
            If indices exceed matrix dimensions.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat.get_values([0,1],[1,2])
        array([1., 1.])
        """
        col_idx = np.array(col_idx)
        row_idx = np.array(row_idx)
        row_idx, col_idx = self._swap_indices(row_idx, col_idx)

        if np.any(row_idx >= self.bin_num) or np.any(col_idx >= self.bin_num):
            raise ValueError("row_idx and col_idx must be less than bin_num")
        # Check if indices are in range
        is_in_range = (col_idx - row_idx) < self.diag_num
        if np.any(~is_in_range):
            warnings.warn(
                "Some indices are out of range for the specified diag_num. "
                "These entries will be set to `default_value` of this `band_hic_matrix` object.",
                UserWarning,
            )
        # Prepare result array
        query_result = np.full(
            shape=(row_idx.shape[0],),
            fill_value=self.default_value,
            dtype=self.dtype,
        )
        # Calculate diagonal offsets
        k_arr = col_idx[is_in_range] - row_idx[is_in_range]
        # Create location array for data access
        loc_arr = (row_idx[is_in_range], k_arr)
        # Retrieve data from matrix
        query_result[is_in_range] = self.data[loc_arr]
        if self.mask is None:
            return query_result  # Return result if no masks are set
        else:
            # Create mask array
            mask = np.zeros_like(query_result, dtype=bool)
            # Set mask for valid entries
            mask[is_in_range] = self.mask[loc_arr]
            # Return masked array
            return ma.MaskedArray(
                query_result, mask=mask, fill_value=self.default_value
            )

    def set_values(
        self,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        values: Union[Number, np.ndarray],
    ) -> None:
        """
        Set values at specified row and column indices.

        Parameters
        ----------
        row_idx : array-like of int
            Row indices where values will be set.
        col_idx : array-like of int
            Column indices where values will be set.
        values : scalar or array-like
            Values to assign.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.zeros((3,3), diag_num=2, dtype=int)
        >>> mat.set_values([0,1], [1,2], [4,5])
        >>> mat[0,1]
        4
        
        Notes
        -----
                Writing to masked positions will update the underlying data but will not clear the mask.
        """
        row_idx = np.array(row_idx)
        col_idx = np.array(col_idx)
        if len(row_idx) != len(col_idx):
            raise ValueError(
                "row_idx and col_idx must have the same length."
            )
        if not isinstance(values, Number) and len(row_idx) != len(values):
            raise ValueError(
                "row_idx, col_idx, and values must have the same length."
            )
        # Swap indices to ensure row_idx <= col_idx
        row_idx, col_idx = self._swap_indices(row_idx, col_idx)
        # Check for valid range
        in_range_idx = (col_idx - row_idx) < self.diag_num
        if np.any(~in_range_idx):
            warnings.warn(
                "Some entries are out of range for the specified diag_num. "
                "These entries will be ignored.",
                UserWarning,
            )
        # Calculate diagonal offsets
        k_arr = col_idx[in_range_idx] - row_idx[in_range_idx]
        # Create location array for data access
        loc_arr = (row_idx[in_range_idx], k_arr)
        if isinstance(values, (int, float, np.integer, np.floating)):
            self.data[loc_arr] = values
        else:
            values = np.array(values)
            self.data[loc_arr] = values[in_range_idx]
        if self.mask is not None:
            if np.any(self.mask[loc_arr]):
                warnings.warn(
                    "Some entries are masked; these will be updated to the new values but will not be unmasked.",
                    UserWarning,
                )
            if self.mask_row_col is not None:
                if np.any(
                    self.mask_row_col[row_idx[in_range_idx]]
                ) or np.any(self.mask_row_col[col_idx[in_range_idx]]):
                    warnings.warn(
                        "Some rows/columns are masked; these will not be unmasked.",
                        UserWarning,
                    )

    def diag(self, k: int) -> Union[np.ndarray, ma.MaskedArray]:
        """
        Retrieve the k-th diagonal from the matrix.

        Parameters
        ----------
        k : int
            The diagonal index to retrieve.

        Returns
        -------
        ndarray or MaskedArray
            The k-th diagonal values; masked if mask is set.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat.diag(1)
        array([1., 1.])
        """
        diag_len = self.shape[0] - k  # Length of the diagonal
        result = self.data[:diag_len, k]  # Copy the diagonal values
        if self.mask is not None:
            # Create mask for the diagonal
            mask = self.mask[:diag_len, k]
            result = ma.MaskedArray(
                result, mask=mask, fill_value=self.default_value
            )
        return result

    def set_diag(self, k: int, values: Iterable[Any]) -> None:
        """
        Set values in the k-th diagonal of the matrix.

        Parameters
        ----------
        k : int
            The diagonal index to set.
        values : array-like
            The values to set in the diagonal.

        Raises
        ------
        ValueError
            If k is out of range or values length mismatch.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.zeros((4,4), diag_num=3)
        >>> mat.set_diag(1, [9,9,9])
        >>> mat.diag(1)
        array([9., 9., 9.])
        """
        diag_num = self.shape[0] - k  # Length of the diagonal
        if k < 0 or k >= self.diag_num:
            raise ValueError("k must be between 0 and diag_num-1")
        if isinstance(values, (list, np.ndarray)):
            if len(values) != diag_num:
                raise ValueError(
                    "Length of values must match the diagonal length"
                )
        elif not isinstance(values, (Number, np.integer, np.floating)):
            raise ValueError("values must be number/list/np.ndarray.")
        self.data[:diag_num, k] = values  # Assign values to the diagonal

    def todense(self) -> Union[np.ndarray, ma.MaskedArray]:
        """
        Convert the band matrix to a dense format.

        Returns
        -------
        ndarray or MaskedArray
            The dense (square) matrix. Masked if mask is set.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> dense = mat.todense()
        >>> dense.shape
        (3, 3)
        """
        result = np.full(
            shape=(self.shape[0], self.shape[1]),
            fill_value=self.default_value,
            dtype=self.dtype,
        )
        # Vectorized filling of the symmetric band matrix
        rows = np.arange(self.bin_num).reshape(-1, 1)  # shape: (N, 1)
        offsets = np.arange(self.diag_num).reshape(1, -1)  # shape: (1, D)
        row_idx = np.repeat(rows, self.diag_num, axis=1)
        col_idx = row_idx + offsets
        valid = col_idx < self.bin_num
        row_idx_valid = row_idx[valid]
        col_idx_valid = col_idx[valid]
        data_vals = self.data[row_idx_valid, col_idx_valid - row_idx_valid]
        result[row_idx_valid, col_idx_valid] = data_vals
        result[col_idx_valid, row_idx_valid] = data_vals  # symmetric
        if self.mask is not None:
            # Create mask for the dense matrix
            mask = np.zeros_like(result, dtype=bool)
            mask_vals = self.mask[
                row_idx_valid, col_idx_valid - row_idx_valid
            ]
            mask[row_idx_valid, col_idx_valid] = mask_vals
            mask[col_idx_valid, row_idx_valid] = mask_vals
            # Create masked array
            if self.mask_row_col is not None:
                mask[:, self.mask_row_col] = True
                mask[self.mask_row_col, :] = True
            result = ma.MaskedArray(
                result, mask=mask, fill_value=self.default_value
            )
        return result

    def tocoo(self, drop_zeros:bool = True) -> coo_array:
        """
        Convert the matrix to COO format.
        
        Parameters
        ----------
        drop_zeros : bool, optional
            If True, zero entries will be dropped from the COO format.
            Default is True.

        Returns
        -------
        coo_array
            The matrix in scipy COO sparse format.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> coo = mat.tocoo()
        >>> coo.shape
        (3, 3)
        """
        row_idx = np.repeat(
            np.arange(0, self.bin_num).reshape(-1, 1), self.diag_num, axis=1
        )
        col_idx = row_idx + np.arange(0, self.diag_num)
        is_valid = ~self.mask if self.mask is not None else col_idx < self.bin_num
        coo_result = coo_array(
            (self.data[is_valid], (row_idx[is_valid], col_idx[is_valid])),
            shape=(self.bin_num, self.bin_num),
            dtype=self.dtype,
        )
        if drop_zeros:
            coo_result.eliminate_zeros()  # Drop zero entries if specified
        return coo_result  # Return COO format

    def tocsr(self) -> csr_array:
        """
        Convert the matrix to CSR format.

        Returns
        -------
        csr_array
            The matrix in scipy CSR sparse format.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> csr = mat.tocsr()
        >>> csr.shape
        (3, 3)
        """
        return self.tocoo(drop_zeros=False).tocsr()  # Convert to CSR format

    def copy(self) -> "band_hic_matrix":
        """
        Deep copy the object.

        Returns
        -------
        band_hic_matrix
            A deep copy.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat2 = mat.copy()
        """
        copy_obj = copy.deepcopy(self)
        return copy_obj

    def memory_usage(self) -> int:
        """
        Compute memory usage of `band_hic_matirx` object.

        Returns
        -------
        int
            Size in bytes.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat.memory_usage()
        772
        """
        total_bytes = (
            sys.getsizeof(self.data)
            + sys.getsizeof(self.mask)
            + sys.getsizeof(self.bin_num)
            + sys.getsizeof(self.diag_num)
            + sys.getsizeof(self.dtype)
            + sys.getsizeof(self.shape)
            + sys.getsizeof(self.default_value)
            + sys.getsizeof(self.mask_row_col)
        )
        return total_bytes

    def astype(self, dtype: type, copy: bool=False) -> "band_hic_matrix":
        """
        Cast data to new dtype.

        Parameters
        ----------
        type : data-type
            Target dtype.
        copy: bool
            If True, the operation is performed in place. Default is False.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat = mat.astype(np.float32)
        """
        if not copy:
            self.data = self.data.astype(dtype)
            self.dtype = dtype  # Update the dtype attribute
            return self
        else:
            return self.copy().astype(dtype,copy=True)

    def clip(self, min_val: Number, max_val: Number) -> None:
        """
        Clip data values to given range.

        Parameters
        ----------
        min_val : scalar
        max_val : scalar

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat = mat.clip(0, 10)
        """
        self.data = np.clip(self.data, min_val, max_val)
        self.default_value = np.clip(self.default_value, min_val, max_val)
        self.dtype = self.data.dtype  # Update dtype if changed
        return self

    def __repr__(self):
        """
        Return a string representation of the band_hic_matrix object.

        Returns
        -------
        str
            A string representation of the object.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> repr(mat)
            "band_hic_matrix(shape=(3, 3), diag_num=2, dtype=<class 'numpy.float64'>)"
        """
        return f"band_hic_matrix(shape={self.shape}, diag_num={self.diag_num}, dtype={self.dtype})"

    def __str__(self):
        """
        Return a string representation of the band_hic_matrix object.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> print(mat)
            band_hic_matrix(shape=(3, 3), diag_num=2, dtype=<class 'numpy.float64'>)
        """
        return f"band_hic_matrix(shape={self.shape}, diag_num={self.diag_num}, dtype={self.dtype})"

    def __len__(self):
        """
        Return the number of rows in the band_hic_matrix object.

        Returns
        -------
        int
            Number of rows.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> len(mat)
        3
        """
        return self.shape[0]

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over diagonals of the matrix.

        Yields
        ------
        ndarray
            Values of each diagonal.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> for band in mat:
        ...     print(band)
        [1. 1. 1.]
        [1. 1.]
        """
        for k in range(self.diag_num):
            yield self.diag(k)

    def iterwindows(
        self, width: int, step: int = 1
    ) -> Iterator["band_hic_matrix"]:
        """
        Iterate over the diagonals of the matrix with a specified window size.

        Parameters
        ----------
        width : int
            The size of the window to iterate over.
        step : int, optional
            Step size between windows. Default is 1.

        Yields
        ------
        band_hic_matrix
            The values in the current window.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> for win in mat.iterwindows(2):
        ...     print(win)
        band_hic_matrix(shape=(2, 2), diag_num=2, dtype=<class 'numpy.float64'>)
        band_hic_matrix(shape=(2, 2), diag_num=2, dtype=<class 'numpy.float64'>)
        """
        if width > self.diag_num or width <= 0:
            raise ValueError(
                "Width must be positive and less than the number of diags."
            )
        if step <= 0:
            raise ValueError("Step must be positive.")
        for i in range(0, self.bin_num - width + 1, step):
            yield self[i : i + width]

    def iterrows(self) -> Iterator[np.ndarray]:
        """
        Iterate over the rows of the band_hic_matrix object.

        Yields
        ------
        ndarray
            The values in the current row.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> for row in mat.iterrows():
        ...     print(row)
        [1. 1. 1.]
        [1. 1. 1.]
        [1. 1. 1.]
        """
        for i in range(self.bin_num):
            # Extract and yield the full row band (symmetric)
            yield self.extract_row(i, extract_out_of_band=True)

    def itercols(self) -> Iterator[np.ndarray]:
        """
        Iterate over the columns of the band_hic_matrix object.

        Yields
        ------
        ndarray
            The values in the current column.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> for col in mat.itercols():
        ...     print(col)
        [1. 1. 1.]
        [1. 1. 1.]
        [1. 1. 1.]
        """
        for i in range(self.bin_num):
            # Columns mirror rows due to symmetry
            yield self.extract_row(i, extract_out_of_band=True)

    def dump(self, filename: str) -> None:
        """
        Save the band_hic_matrix object to a file.

        Parameters
        ----------
        filename : str
            The name of the file to save the object to.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.ones((3,3), diag_num=2)
        >>> mat.dump('myfile.npz')
        """
        np.savez(
            filename,
            data=self.data,
            dtype=self.dtype,
            mask=self.mask,
            mask_row_col=self.mask_row_col,
            default_value=self.default_value,
        )

    def extract_row(
        self, idx: int, extract_out_of_band: bool = True
    ) -> Union[np.ndarray, ma.MaskedArray]:
        """
        Extract stored, unmasked band values for a given row or column.

        Parameters
        ----------
        idx : int
            Row (or column, due to symmetry) index for which to extract band values.
        extract_out_of_band : bool, optional
            If True, include out-of-band entries filled with `default_value` and masked if appropriate.
            If False (default), return only the stored band values.

        Returns
        -------
        ndarray or MaskedArray
            If `extract_out_of_band=False`, a 1D array of length up to `diag_num` containing band values.
            If `extract_out_of_band=True`, a 1D array of length `bin_num` with all row/column values.

        Raises
        ------
        ValueError
            If `idx` is outside the range [0, bin_num-1].

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.extract_row(0, extract_out_of_band=False)
        array([0, 1])
        >>> mat.extract_row(0)
        array([0, 1, 0])
        """
        if self.mask_row_col is not None and self.mask_row_col[idx]:
            # If the row/column is masked, return a masked array
            if extract_out_of_band:
                return ma.MaskedArray(
                    np.full(
                        self.bin_num,
                        fill_value=self.default_value,
                        dtype=self.dtype,
                    ),
                    mask=True,
                    fill_value=self.default_value,
                )
            else:
                item_num = min(self.bin_num, idx + self.diag_num) - max(
                    0, idx - self.diag_num + 1
                )
                return ma.MaskedArray(
                    np.full(
                        item_num,
                        fill_value=self.default_value,
                        dtype=self.dtype,
                    ),
                    mask=True,
                    fill_value=self.default_value,
                )
        n = self.bin_num
        d = self.diag_num
        # forward diagonal entries
        k_f = min(d, n - idx)
        vals_f = self.data[idx, :k_f]
        if self.mask is not None:
            # Create mask for the forward diagonal
            mask_f = self.mask[idx, :k_f]
            vals_f = ma.MaskedArray(
                vals_f, mask=mask_f, fill_value=self.default_value
            )
        first_row = max(0, idx - d + 1)
        last_row = idx
        row_range = np.arange(first_row, last_row)
        vals_b = self.data[row_range, idx - row_range]
        if self.mask is not None:
            # Create mask for the backward diagonal
            mask_b = self.mask[row_range, idx - row_range]
            vals_b = ma.MaskedArray(
                vals_b, mask=mask_b, fill_value=self.default_value
            )
        # Concatenate forward and backward diagonal values
        if not extract_out_of_band:
            if self.mask is not None:
                # Create mask for the concatenated values
                mask = np.concatenate([mask_b, mask_f])
                vals = np.concatenate([vals_b.data, vals_f.data])
                vals = ma.MaskedArray(
                    vals, mask=mask, fill_value=self.default_value
                )
            else:
                vals = np.concatenate([vals_b, vals_f])
            # Return the concatenated values
            return vals
        else:
            vals_row = np.full(
                n, dtype=self.dtype, fill_value=self.default_value
            )
            if self.mask is not None:
                vals_row[first_row:last_row] = vals_b.data
                vals_row[idx : idx + k_f] = vals_f.data
                mask_row = np.zeros(n, dtype=bool)
                mask_row[first_row:last_row] = mask_b
                mask_row[idx : idx + k_f] = mask_f
                if self.mask_row_col is not None:
                    mask_row[self.mask_row_col] = True
                vals_row = ma.MaskedArray(
                    vals_row, mask=mask_row, fill_value=self.default_value
                )
            else:
                vals_row[first_row:last_row] = vals_b
                vals_row[idx : idx + k_f] = vals_f
            return vals_row

    def min(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the minimum value in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the minimum:
            - None: compute over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Minimum value(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.min()  # global minimum
        0
        >>> mat.min(axis='row')
        array([0, 1, 0])
        """
        # Normalize axis: accept 0/1 or 'row'/'col'
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out, mask=False, fill_value=self.default_value
                )
                if self.mask_row_col is not None:
                    out[self.mask_row_col] = ma.masked
            for i in range(self.bin_num):
                if self.mask_row_col is None or (
                    self.mask_row_col is not None and not self.mask_row_col[i]
                ):
                    row_vals = self.extract_row(i, extract_out_of_band=True)
                    out[i] = row_vals.min()
            return out
        elif axis == "diag":
            out = np.empty(self.diag_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out, mask=False, fill_value=self.default_value
                )
            for k in range(self.diag_num):
                out[k] = self.diag(k).min()
            return out
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data[~self._extract_raw_mask(self.data.shape)]
            )
            if flat.size == 0:
                return ma.masked
            if self.bin_num == self.diag_num:
                return flat.min()
            else:
                all_min = flat.min()
                return (
                    self.default_value
                    if all_min is ma.masked
                    else min(all_min, self.default_value)
                )
        else:
            raise ValueError(
                "Unsupported axis for min: choose 0,1,'row','col','diag', or None"
            )

    def max(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the maximum value in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the maximum:
            - None: compute over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Maximum value(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.max()  # global maximum
        8
        >>> mat.max(axis='row')
        array([1, 5, 8])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out, mask=False, fill_value=self.default_value
                )
                if self.mask_row_col is not None:
                    out[self.mask_row_col] = ma.masked
            for i in range(self.bin_num):
                if self.mask_row_col is None or (
                    self.mask_row_col is not None and not self.mask_row_col[i]
                ):
                    row_vals = self.extract_row(i, extract_out_of_band=True)
                    out[i] = row_vals.max()
            return out
        elif axis == "diag":
            out = np.empty(self.diag_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out, mask=False, fill_value=self.default_value
                )
            for k in range(self.diag_num):
                out[k] = self.diag(k).max()
            return out
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data[~self._extract_raw_mask(self.data.shape)]
            )
            if flat.size == 0:
                return self.default_value
            elif self.bin_num == self.diag_num:
                return flat.max()
            else:
                return max(flat.max(), self.default_value)
        else:
            raise ValueError(
                "Unsupported axis for max: choose 0,1,'row','col','diag', or None"
            )

    def sum(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the sum of the values in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the sum:
            - None: sum over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Sum(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.sum()  # sum of all elements
        24
        >>> mat.sum(axis='row')
        array([ 1, 10, 13])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out,
                    mask=False,
                    fill_value=self.default_value,
                )
                if self.mask_row_col is not None:
                    out[self.mask_row_col] = ma.masked

            for i in range(self.bin_num):
                if self.mask_row_col is None or (
                    self.mask_row_col is not None and not self.mask_row_col[i]
                ):
                    band = self.extract_row(i, extract_out_of_band=True)
                    out[i] = band.sum()
            return out
        elif axis == "diag":
            if self.mask is not None:
                return ma.array(
                    [self.diag(k).sum() for k in range(self.diag_num)],
                    mask=False,
                    fill_value=self.default_value,
                )
            else:
                return np.array(
                    [self.diag(k).sum() for k in range(self.diag_num)]
                )
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data[~self._extract_raw_mask(self.data.shape)]
            )
            return (
                flat.sum() * 2
                - self.diag(0).sum()
                + self.default_value
                * (self.count_out_band() - self.count_out_band_masked())
            )
        else:
            raise ValueError(
                "Unsupported axis for sum: choose 0,1,'row','col','diag', or None"
            )

    def mean(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the mean value of the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the mean:
            - None: mean over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Mean value(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.mean()  # mean of all elements
        2.6666666666666665
        >>> mat.mean(axis='diag')
        array([4., 3.])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            result = np.zeros(self.bin_num, dtype=np.float64)
            if self.mask is not None:
                result = ma.MaskedArray(
                    result,
                    mask=False,
                    fill_value=self.default_value,
                )
                if self.mask_row_col is not None:
                    result[self.mask_row_col] = ma.masked
            for i in range(self.bin_num):
                if self.mask_row_col is None or (
                    self.mask_row_col is not None and not self.mask_row_col[i]
                ):
                    band = self.extract_row(i, extract_out_of_band=True)
                    result[i] = band.mean()
            return result
        elif axis == "diag":
            if self.mask is not None:
                return ma.array(
                    [self.diag(k).mean() for k in range(self.diag_num)],
                    mask=False,
                    fill_value=self.default_value,
                )
            else:
                # Compute mean for each diagonal
                # If no values, return default_value
                return np.array(
                    [self.diag(k).mean() for k in range(self.diag_num)]
                )
        elif axis is None:
            return self.sum() / self.count_unmasked()
        else:
            raise ValueError(
                "Unsupported axis for sum: choose 0,1,'row','col','diag', or None"
            )

        # Elementwise mean: for None returns scalar, for diag/row returns array

    def prod(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the product of the values in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the product:
            - None: product over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Product(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(1, 10).reshape(3,3), diag_num=2)
        >>> mat.prod()  # product of all elements
        0
        >>> mat.prod(axis='row')
        array([ 0, 60, 0])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=self.dtype)
            if self.mask is not None:
                out = ma.MaskedArray(
                    out,
                    mask=False,
                    fill_value=self.default_value,
                )
                if self.mask_row_col is not None:
                    out[self.mask_row_col] = ma.masked
            for i in range(self.bin_num):
                if self.mask_row_col is None or (
                    self.mask_row_col is not None and not self.mask_row_col[i]
                ):
                    band = self.extract_row(i, extract_out_of_band=False)
                    missing = self.bin_num - band.size
                    prod_val = band.prod() if band.size > 0 else 1
                    out[i] = prod_val * (self.default_value**missing)
            return out
        elif axis == "diag":
            if self.mask is not None:
                return ma.array(
                    [self.diag(k).prod() for k in range(self.diag_num)],
                    mask=False,
                    fill_value=self.default_value,
                )
            else:
                return np.array(
                    [self.diag(k).prod() for k in range(self.diag_num)]
                )
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data.ravel()
            )
            missing = self.bin_num * self.bin_num - flat.size
            prod_val = flat.prod() if flat.size > 0 else 1
            return prod_val * (self.default_value**missing)
        else:
            raise ValueError(
                "Unsupported axis for prod: choose 0,1,'row','col','diag', or None"
            )

    def var(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the variance of the values in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the variance:
            - None: variance over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Variance(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.var()  # variance of all elements
        7.555555555555555
        >>> mat.var(axis='row')
        array([ 0.22222222,  2.88888889, 10.88888889])
        """
        # variance = mean(x^2) - mean(x)^2
        m = self.mean(axis=axis)
        # Sum of squares along axis
        sum_sq = self.square().sum(axis=axis)
        # Determine counts based on axis
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            if self.mask is not None:
                counts = np.array(
                    [
                        self.extract_row(i, extract_out_of_band=True).count()
                        for i in range(self.bin_num)
                    ]
                )
            else:
                counts = self.bin_num
        elif axis == "diag":
            if self.mask is not None:
                counts = np.array(
                    [self.diag(k).count() for k in range(self.diag_num)]
                )
            else:
                counts = np.array(
                    [self.bin_num - k for k in range(self.diag_num)]
                )
        elif axis is None:
            counts = self.count_unmasked()
        else:
            raise ValueError(
                "Unsupported axis for var: choose 0,1,'row','col','diag', or None"
            )
        return sum_sq / counts - m**2

    def std(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the standard deviation of the values in the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the standard deviation:
            - None: std over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Standard deviation(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.std()  # std of all elements
        2.748737083745107
        >>> mat.std(axis='diag')
        array([3.26598632, 2.        ])
        """
        # standard deviation = sqrt(var)
        return np.sqrt(self.var(axis=axis))

    def normalize(self, inplace: bool = False) -> None:
        """
        Normalize each diagonal of the matrix to have zero mean and unit variance.
        This modifies the matrix in place.
        Raises
        ------
        UserWarning
            If any diagonal has zero standard deviation, it will be set to zero.
        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat = mat.normalize()
        """
        if inplace:
            result = self
        else:
            result = self.copy()
        if result.dtype != np.float64:
            result.data = result.data.astype(
                np.float64
            )  # Ensure data is float for normalization
            result.dtype = np.float64
        for k in range(self.diag_num):
            diag = self.diag(k)
            mean = diag.mean()
            std = diag.std()
            if std == 0:
                warnings.warn(
                    f"Diagonal {k} has zero standard deviation, set all values to zero.".format(
                        k
                    ),
                    UserWarning,
                )
                # If std is zero, set all values to zero
                result.set_diag(k, 0)
            else:
                result.set_diag(k, (diag - mean) / std)
        if not inplace:
            return result
        else:
            return None

    def ptp(
        self, axis: Optional[Union[int, str]] = None
    ) -> Union[Number, np.ndarray]:
        """
        Compute the peak-to-peak (maximum - minimum) value of the matrix or along a given axis.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to compute the peak-to-peak value:
            - None: ptp over all stored values (and default for missing).
            - 0 or 'row': per-row reduction.
            - 1 or 'col': per-column reduction.
            - 'diag': per-diagonal reduction.
            Default is None.

        Returns
        -------
        scalar or ndarray
            Peak-to-peak value(s) along the specified axis.

        Raises
        ------
        ValueError
            If axis is not one of the supported values.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.arange(9).reshape(3,3), diag_num=2)
        >>> mat.ptp()  # ptp of all elements
        8
        >>> mat.ptp(axis='row')
        array([1, 4, 8])
        """
        max_vals = self.max(axis=axis)
        min_vals = self.min(axis=axis)
        return max_vals - min_vals

    def all(
        self, axis: Optional[Union[int, str]] = None, banded_only: bool = False
    ) -> Union[np.bool_, np.ndarray[np.bool_, Any]]:
        """
        Test whether all (or any) array elements along a given axis evaluate to True.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to test.
        banded_only : bool, optional
            If True, only consider stored band elements; ignore out-of-band values.
            Default is False.

        Returns
        -------
        bool or ndarray
            Boolean result(s) of the test.

        Raises
        ------
        ValueError
            If axis is not supported.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> mat.all()
        False
        >>> mat.any(axis='diag', banded_only=True)
        array([ True, False])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=bool)
            if banded_only:
                for i in range(self.bin_num):
                    band = self.extract_row(i, extract_out_of_band=False)
                    if self.mask is not None:
                        band = band.compressed()
                    out[i] = np.all(band)
                return out
            else:
                for i in range(self.bin_num):
                    band = self.extract_row(i, extract_out_of_band=False)
                    missing = self.bin_num - band.size
                    if self.mask is not None:
                        band = band.compressed()
                    out[i] = np.all(band)
                    if missing > 0:
                        out[i] = out[i] and self.default_value
                return out
        elif axis == "diag":
            if banded_only:
                if self.mask is not None:
                    return np.array(
                        [
                            np.all(self.diag(k).compressed())
                            for k in range(self.diag_num)
                        ]
                    )
                else:
                    return np.array(
                        [np.all(self.diag(k)) for k in range(self.diag_num)]
                    )
            else:
                missing = self.bin_num - self.diag_num
                if self.mask is not None:
                    out = np.array(
                        [
                            np.all(self.diag(k).compressed())
                            for k in range(self.diag_num)
                        ]
                    )
                    out = np.concatenate(
                        [out, np.full(missing, self.default_value)]
                    )
                else:
                    out = np.array(
                        [np.all(self.diag(k)) for k in range(self.diag_num)]
                    )
                    out = np.concatenate(
                        [out, np.full(missing, self.default_value)]
                    )
                return out
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data.ravel()
            )
            if banded_only:
                return np.all(flat)
            else:
                if self.diag_num < self.bin_num:
                    return np.all(flat) and bool(self.default_value)
                else:
                    return np.all(flat)
        else:
            raise ValueError(
                "Unsupported axis for all: choose 0,1,'row','col','diag', or None"
            )

    def any(
        self, axis: Optional[Union[int, str]] = None, banded_only: bool = False
    ) -> Union[bool, np.ndarray]:
        """
        Test whether all (or any) array elements along a given axis evaluate to True.

        Parameters
        ----------
        axis : None, int, or {'row','col','diag'}, optional
            Axis along which to test.
        banded_only : bool, optional
            If True, only consider stored band elements; ignore out-of-band values.
            Default is False.

        Returns
        -------
        bool or ndarray
            Boolean result(s) of the test.

        Raises
        ------
        ValueError
            If axis is not supported.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> mat.all()
        False
        >>> mat.any(axis='diag', banded_only=True)
        array([ True, False])
        """
        if axis in (1, "col"):
            axis = 0
        if axis in (0, "row"):
            out = np.empty(self.bin_num, dtype=bool)
            if banded_only:
                for i in range(self.bin_num):
                    band = self.extract_row(i, extract_out_of_band=False)
                    if self.mask is not None:
                        band = band.compressed()
                    out[i] = np.any(band)
                return out
            else:
                for i in range(self.bin_num):
                    band = self.extract_row(i, extract_out_of_band=False)
                    missing = self.bin_num - band.size
                    if self.mask is not None:
                        band = band.compressed()
                    out[i] = np.any(band)
                    if missing > 0:
                        out[i] = out[i] or self.default_value
                return out
        elif axis == "diag":
            if banded_only:
                if self.mask is not None:
                    return np.array(
                        [
                            np.any(self.diag(k).compressed())
                            for k in range(self.diag_num)
                        ]
                    )
                else:
                    return np.array(
                        [np.any(self.diag(k)) for k in range(self.diag_num)]
                    )
            else:
                missing = self.bin_num - self.diag_num
                if self.mask is not None:
                    out = np.array(
                        [
                            np.any(self.diag(k).compressed())
                            for k in range(self.diag_num)
                        ]
                    )
                    out = np.concatenate(
                        [out, np.full(missing, self.default_value)], dtype=bool
                    )
                else:
                    out = np.array(
                        [np.any(self.diag(k)) for k in range(self.diag_num)]
                    )
                    out = np.concatenate(
                        [out, np.full(missing, self.default_value)], dtype=bool
                    )
                return out
        elif axis is None:
            flat = (
                self.data[~self.mask]
                if self.mask is not None
                else self.data.ravel()
            )
            if banded_only:
                return np.any(flat)
            else:
                if self.diag_num < self.bin_num:
                    return np.any(flat) or self.default_value
                else:
                    return np.any(flat)
        else:
            raise ValueError(
                "Unsupported axis for any: choose 0,1,'row','col','diag', or None"
            )

    def __contains__(self, item):
        """
        Check whether a value exists in the band_hic_matrix.

        Parameters
        ----------
        item : scalar
            Value to check for membership in the matrix (including masked values if present).

        Returns
        -------
        bool
            True if `item` is present in the stored matrix (ignoring masked entries), False otherwise.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> 1 in mat
        True
        >>> 99 in mat
        False
        """
        arr = self.data
        if self.mask is not None:
            mask = self.mask
        else:
            mask = self._extract_raw_mask(self.data.shape)
        return np.any(arr[~mask] == item) or self.default_value == item

    def __hash__(self):
        """
        Return a hash value for the band_hic_matrix object.

        Returns
        -------
        int
            Hash value.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> hashed=hash(mat)
        """
        if self.mask is not None:
            # Include mask in hash if present
            return hash(
                (
                    self.shape,
                    self.diag_num,
                    self.dtype,
                    tuple(self.data.flatten()),
                    tuple(self.mask.flatten()),
                )
            )
        else:
            return hash(
                (
                    self.shape,
                    self.diag_num,
                    self.dtype,
                    tuple(self.data.flatten()),
                )
            )

    def __bool__(self) -> bool:
        """
        Truth value of the band_hic_matrix, following NumPy semantics.

        Returns
        -------
        bool
            If the matrix has exactly one element (shape (1,1)), returns its truth value;
            otherwise raises a ValueError.

        Raises
        ------
        ValueError
            If the matrix contains more than one element.
        """
        # Only a 1x1 matrix can be truth-tested
        if self.shape != (1, 1):
            raise ValueError(
                "The truth value of a band_hic_matrix with more than one element is ambiguous. "
                "Use a.any() or a.all()."
            )
        # Single element at data[0,0]; consider mask if present
        if self.mask is not None and self.mask[0, 0]:
            return False
        return bool(self.data[0, 0])

    def __array__(self, copy=False):
        """
        Return the data as a NumPy array.

        Parameters
        ----------
        copy : bool, optional
            If True, returns a copy. Default is False.

        Returns
        -------
        ndarray
            Underlying band data array.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> arr = np.array(mat)
        """
        return self.data

    def __array_priority__(self):
        """
        Return the priority for array operations.

        Returns
        -------
        int
            Priority value.

        Examples
        --------
        >>> import bandhic as bh
        >>> mat = bh.band_hic_matrix(np.eye(3), diag_num=2)
        >>> mat.__array_priority__()
        100
        """
        return 100

    def _ufunc_handle_out(self, kwargs):
        """
        Handle 'out' kwarg for ufunc, converting band_hic_matrix outputs to ndarrays.

        Parameters
        ----------
        kwargs : dict
            Keyword args passed to ufunc.

        Returns
        -------
        band_outs : list or None
            List of band_hic_matrix instances if out provided, otherwise None.
        """
        outs = kwargs.get("out", None)
        band_outs = None
        if outs is not None:
            band_outs = []
            new_outs = []
            for o in outs:
                if isinstance(o, band_hic_matrix):
                    band_outs.append(o)
                    new_outs.append(o.data)
                else:
                    raise ValueError("Output must be band_hic_matrix or None.")
            kwargs["out"] = tuple(new_outs)
        return band_outs

    def _ufunc_prepare_inputs(self, inputs):
        """
        Prepare ufunc inputs: extract data arrays, masks, and validate shapes.

        Parameters
        ----------
        inputs : tuple
            Positional args passed to ufunc.

        Returns
        -------
        inputs_data : list
        mask_list : list
        is_masked : bool
        shape : tuple
        diag_num : int
        """
        inputs_data = []
        mask_list = []
        mask_row_col_list = []
        is_masked = False
        shape = None
        diag_num = None
        defaults = []
        for inp in inputs:
            if isinstance(inp, band_hic_matrix):
                if shape is not None and shape != inp.shape:
                    raise ValueError(
                        "Shapes of band_hic_matrix objects do not match."
                    )
                if diag_num is not None and diag_num != inp.diag_num:
                    raise ValueError(
                        "Diagonal numbers of band_hic_matrix objects do not match."
                    )
                shape = inp.shape
                diag_num = inp.diag_num
                inputs_data.append(inp.data)
                if inp.mask is not None:
                    is_masked = True
                    mask_list.append(inp.mask)
                    if inp.mask_row_col is not None:
                        mask_row_col_list.append(inp.mask_row_col)
                defaults.append(inp.default_value)
            elif isinstance(inp, (Number, np.integer, np.floating)):
                inputs_data.append(inp)
                defaults.append(inp)
            else:
                raise ValueError("Inputs must be band_hic_matrix or number.")
        if shape is None:
            raise ValueError("No valid input shapes found.")
        if diag_num is None:
            raise ValueError("No valid input diagonal numbers found.")
        return (
            inputs_data,
            mask_list,
            mask_row_col_list,
            is_masked,
            shape,
            diag_num,
            defaults,
        )

    def _ufunc_apply_mask_where(self, kwargs, mask_list):
        """
        Combine input masks and adjust 'where' kwarg for ufunc.

        Parameters
        ----------
        kwargs : dict
        mask_list : list of ndarray
        """
        combined_mask = np.logical_or.reduce(mask_list)
        orig_where = kwargs.get("where", None)
        if orig_where is None:
            kwargs["where"] = ~combined_mask
        else:
            kwargs["where"] = orig_where & (~combined_mask)

    def _ufunc_wrap_results(
        self, results, band_outs, mask_list, mask_row_col_list, results_default
    ):
        """
        Wrap ufunc result arrays back into band_hic_matrix instances.

        Parameters
        ----------
        results : tuple of ndarray
        band_outs : list or None
        mask_list : list of ndarray
        resutls_default : list of Number

        Returns
        -------
        list of band_hic_matrix
        """
        outputs = []
        for idx, arr in enumerate(results):
            if band_outs is not None and idx < len(band_outs):
                obj = band_outs[idx]
                obj.data = arr
            else:
                obj = self.__class__(arr, band_data_input=True)
            obj.dtype = arr.dtype
            if mask_list:
                obj.mask = np.logical_or.reduce(mask_list)
                # obj.mask = np.logical_or(obj.mask, np.isnan(obj.data))
                if mask_row_col_list:
                    obj.mask_row_col = np.logical_or.reduce(mask_row_col_list)
                else:
                    obj.mask_row_col = None
            else:
                obj.mask = None
            obj.default_value = results_default[idx]
            outputs.append(obj)
        return outputs

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> Union["band_hic_matrix", Tuple["band_hic_matrix", ...]]:
        """
        Handle ufunc operations for the band_hic_matrix object.
        """
        # Disallow unsupported ufunc parameters
        for param in ("axis", "keepdims", "dtype"):
            if param in kwargs:
                raise ValueError(
                    f"ufunc parameter '{param}' not supported for band_hic_matrix"
                )
        if method == "__call__":
            band_outs = self._ufunc_handle_out(kwargs)
            (
                inputs_data,
                mask_list,
                mask_row_col_list,
                is_masked,
                shape,
                diag_num,
                defaults,
            ) = self._ufunc_prepare_inputs(inputs)
            if is_masked:
                self._ufunc_apply_mask_where(kwargs, mask_list)
            results = ufunc(*inputs_data, **kwargs)
            results_default = ufunc(*defaults)
            if ufunc.nout == 1:
                results = (results,)
                results_default = (results_default,)
            outputs = self._ufunc_wrap_results(
                results,
                band_outs,
                mask_list,
                mask_row_col_list,
                results_default,
            )
            return outputs[0] if ufunc.nout == 1 else tuple(outputs)
        else:
            return NotImplemented

    def __array_function__(
        self,
        func: Callable,
        types: Tuple[type, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """
        Intercept NumPy top-level API calls and route supported functions to band_hic_matrix methods.

        Parameters
        ----------
        func : function
            NumPy function being called (e.g., np.sum, np.min, np.prod, np.var, np.std, np.ptp, np.all, np.any).
        types : tuple of types
            Types of all arguments provided.
        args : tuple
            Positional arguments passed to func.
        kwargs : dict
            Keyword arguments passed to func. Supports 'axis' and 'out'; other keywords are not supported.

        Returns
        -------
        scalar, ndarray, band_hic_matrix, or tuple
            Result of the function call. If 'out' is provided, writes results in-place and returns the out object or tuple.

        Raises
        ------
        ValueError
            If unsupported keyword arguments (e.g., 'keepdims', 'dtype') are provided.
        """
        # Reject unsupported keyword arguments
        for p in ("keepdims", "dtype"):
            if p in kwargs:
                raise ValueError(
                    f"Keyword argument '{p}' is not supported for band_hic_matrix."
                )

        # Only process if all argument types are supported
        if not all(
            issubclass(t, (band_hic_matrix, np.ndarray, Number)) for t in types
        ):
            return NotImplemented

        # Look up the method name in the registry
        method_name = _ARRAY_FUNCTION_DISPATCH.get(func)
        if method_name is None:
            return NotImplemented
        method = getattr(self, method_name)

        # Call the method: reductions take axis, others take positional args after self
        if method_name in {
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "var",
            "std",
            "ptp",
            "all",
            "any",
        }:
            result = method(axis=kwargs.get("axis", None))
        else:
            # args[0] is self, so skip it
            result = method(*args[1:], **kwargs)

        # Handle out argument
        out = kwargs.get("out", None)
        if out is not None:
            # out may be (arr,) or (arr1, arr2)
            # Handle both single and multiple outputs
            if isinstance(out, tuple):
                # Write result back to out[0] (fill scalar or assign array)
                arr0 = out[0]
                arr0[...] = result
                return out
            else:
                out[...] = result
                return out

        return result