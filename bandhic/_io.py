# -*- coding: utf-8 -*-
# _io.py

"""
_io: Input/Output utilities for BandHiC.
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

Provides functions to save and load band_hic_matrix objects to/from .npz files,
and to read Hi-C data from .hic and .cool files into band_hic_matrix objects.
"""

from .bandhic import band_hic_matrix
import numpy as np
from typing import Dict, Optional
import hicstraw
import cooler

__all__ = [
    "save_npz",
    "load_npz",
    "straw_chr",
    "straw_all_chrs",
    "cooler_chr",
    "cooler_all_chrs",
    "cooler_chr_all_cells",
    "cooler_all_cells_all_chrs",
]

def save_npz(file_name: str, mat: band_hic_matrix) -> None:
    """
    Save a band_hic_matrix to a .npz file.

    Parameters
    ----------
    file_name : str
        Path to save the .npz file.
    mat : band_hic_matrix
        The band_hic_matrix object to save.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.band_hic_matrix(np.eye(5), diag_num=3)
    >>> save_npz('./test/sample.npz', mat)
    """
    np.savez(
        file_name,
        data=mat.data,
        mask=mat.mask,
        mask_row_col=mat.mask_row_col,
        default_value=mat.default_value,
        dtype=mat.dtype,
    )
    # Save the band_hic_matrix to a .npz file

def load_npz(file_name: str) -> band_hic_matrix:
    """
    Load a band_hic_matrix from a .npz file.

    Parameters
    ----------
    file_name : str
        Path to the .npz file.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object loaded from the file.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.load_npz('./test/sample.npz')
    >>> isinstance(mat, band_hic_matrix)
    True
    """
    data = np.load(file_name, allow_pickle=True)
    if (
        "data" not in data
        or "dtype" not in data
        or "default_value" not in data
        or "mask" not in data
        or "mask_row_col" not in data
    ):
        raise ValueError("Invalid .npz file format for band_hic_matrix.")
    if not isinstance(data["data"], np.ndarray):
        raise ValueError("Data in .npz file must be a NumPy ndarray.")
    mask_row_col = data.get("mask_row_col")
    mask = data.get("mask")
    if mask_row_col.dtype != np.bool_:
        _mask_row_col = None
    else:
        _mask_row_col = mask_row_col
    if mask.dtype != np.bool_:
        _mask = None
    else:
        _mask = mask
    return band_hic_matrix(
        data["data"],
        dtype=data["dtype"].item(),
        mask=_mask,
        mask_row_col=_mask_row_col,
        default_value=data["default_value"].item(),
        band_data_input=True,
    )
    
def straw_chr(
    hic_file: str,
    chrom: str,
    resolution: int,
    diag_num: int,
    data_type: str = "observed",
    normalization: str = "NONE",
    unit: str = "BP",
) -> band_hic_matrix:
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
    >>> mat = bh.straw_chr('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', 'chr1', resolution=10000, diag_num=200)
    >>> isinstance(mat, band_hic_matrix)
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

#TODO: can more fast?
    row_idx = np.array(
        [record.binX // resolution for record in records]
    )
    col_idx = np.array(
        [record.binY // resolution for record in records]
    )
    coo_data = np.array([record.counts for record in records])

    mat = band_hic_matrix(
        (coo_data, (row_idx, col_idx)), diag_num=diag_num
    )
    # Set the mask for invalid rows and columns
    return mat

def straw_all_chrs(
    hic_file: str,
    resolution: int,
    diag_num: int,
    data_type: str = "observed",
    normalization: str = "NONE",
    unit: str = "BP",
) -> Dict[str, band_hic_matrix]:
    """
    Read Hi-C data from a .hic file for all chromosomes and return a dictionary of band_hic_matrix objects.

    Parameters
    ----------
    hic_file : str
        Path to the .hic file. This file should be in the Hi-C format compatible with hicstraw. Local or remote paths are supported.
    resolution : int
        Resolution of the Hi-C data. Such as 10000 for 10kb resolution.
    diag_num : int
        Number of diagonals to consider.
    data_type : str, optional
        Type of data to read from the Hi-C file. Default is 'observed'. Other options include 'expected', 'balanced', etc.
        See `hicstraw` documentation for more details.
    normalization : str, optional
        Normalization method to apply. Default is 'NONE'. Other options include 'VC', 'VC_SQRT', 'KR', 'SCALE', etc.
        See `hicstraw` documentation for more details.
    unit : str, optional
        Unit of measurement for the Hi-C data. Default is 'BP' (base pairs). Other options include 'FRAG' (fragments), etc.

    Returns
    -------
    Dict[str, band_hic_matrix]
        A dictionary mapping chromosome names to band_hic_matrix objects containing the Hi-C data.

    Raises
    ------
    ValueError
        If the file cannot be parsed or parameters are invalid.

    Examples
    --------
    >>> import bandhic as bh
    >>> mats = bh.straw_all_chrs('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', resolution=10000, diag_num=200)
    >>> isinstance(mats['chr1'], band_hic_matrix)
    True
    """
    chroms = hicstraw.HiCFile(hic_file).getChromosomes()
    mats = {}
    
    for chrom in chroms:
        chrom = chrom.name
        if chrom == "ALL" or chrom == "M":
            continue
        chrom_long = chrom if chrom.startswith("chr") else f"chr{chrom}"
        mats[chrom_long] = straw_chr(
            hic_file,
            chrom_long,
            resolution,
            diag_num,
            data_type=data_type,
            normalization=normalization,
            unit=unit,
        )
    
    return mats

def cooler_chr(
    file_path: str,
    chrom: str,
    diag_num: int,
    cell_id: Optional[str] = None,
    resolution: Optional[int] = None,
    balance: bool = True,
) -> band_hic_matrix:
    """
    Read Hi-C data from a .cool or .mcool file and return a band_hic_matrix.

    Parameters
    ----------
    file_path : str
        Path to the .cool, .mcool or .scool file.
    chrom : str
        Chromosome name.
    diag_num : int
        Number of diagonals to consider.
    cell_id : str, optional
        Cell ID for .scool files.
    resolution : int, optional
        Resolution of the Hi-C data. 
    balance : bool, optional
        If True, use balanced data. Default is False. This parameter is specific to cooler files.

    Returns
    -------
    band_hic_matrix
        A band_hic_matrix object containing the Hi-C data.

    Raises
    ------
    ValueError
        If the cooler file is invalid or parameters are incorrect.

    Examples
    --------
    >>> import bandhic as bh
    >>> mat = bh.cooler_chr('/Users/wwb/Documents/workspace/BandHiC-Master/data/yeast.10kb.cool', 'chrI', resolution=10000, diag_num=10)
    >>> isinstance(mat, band_hic_matrix)
    True
    
    See also
    --------
    `cooler` documentation for more details on available parameters and usage.
    URL: https://cooler.readthedocs.io/en/latest/index.html
    """
    file_format = file_path.split(".")[-1].lower()
    if file_format == "mcool":
        if resolution is None:
            raise ValueError("resolution is required for .mcool files")
        # For .scool files, we need to specify the group path
        cool_file = file_path + "::resolutions/{resolution}".format(resolution=resolution)
    elif file_format == "cool":
        # For .cool and .mcool files, we can use the file path directly
        cool_file = file_path
    # TODO: support and test scool files
    elif file_format == "scool":
        cool_file = file_path + "::/cells/{cell_id}".format(cell_id=cell_id)
    else:
        raise ValueError(
            f"Unsupported file format: {file_format}. Supported formats are .cool, .mcool, and .cool."
        )
    try:
        clr = cooler.Cooler(cool_file)
        coo_matrix = clr.matrix(balance=balance, sparse=True).fetch(chrom)
    except Exception as e:
        raise ValueError(
            f"Failed to read cooler group '{cool_file}' for chromosome '{chrom}': {e}, please check the file and parameters."
        )
    mat = band_hic_matrix(contacts=coo_matrix, diag_num=diag_num)
    return mat

def cooler_all_chrs(
    file_path: str,
    diag_num: int,
    resolution: Optional[int] = None,
    cell_id: Optional[str] = None,
    balance: bool = True,
) -> Dict[str, band_hic_matrix]:
    """
    Read Hi-C data from a .cool or .mcool file for all chromosomes and return a dictionary of band_hic_matrix objects.

    Parameters
    ----------
    file_path : str
        Path to the .cool, .mcool or .scool file.
    diag_num : int
        Number of diagonals to consider.
    resolution : int, optional
        Resolution of the Hi-C data. 
    cell_id : str, optional
        Cell ID for .scool files.
    balance : bool, optional
        If True, use balanced data. Default is False. This parameter is specific to cooler files.

    Returns
    -------
    Dict[str, band_hic_matrix]
        A dictionary mapping chromosome names to band_hic_matrix objects containing the Hi-C data.

    Raises
    ------
    ValueError
        If the cooler file is invalid or parameters are incorrect.

    Examples
    --------
    >>> import bandhic as bh
    >>> mats = bh.cooler_all_chrs('/Users/wwb/Documents/workspace/BandHiC-Master/data/yeast.10kb.cool', diag_num=10, resolution=10000)
    >>> isinstance(mats['chrI'], band_hic_matrix)
    True
    """
    clr = cooler.Cooler(file_path)
    
    mats = {}
    
    for chrom in clr.chromnames:
        mats[chrom] = cooler_chr(
            file_path,
            chrom,
            cell_id=cell_id,
            diag_num=diag_num,
            resolution=resolution,
            balance=balance,
        )
    
    return mats

# TODO: need test for scool files
def cooler_chr_all_cells(
    file_path: str,
    chrom: str,
    diag_num: int,
    balance: bool = True,
) -> Dict[str, band_hic_matrix]:
    """
    Read Hi-C data from a .scool file for a specific chromosome and return a dictionary of band_hic_matrix objects for all cells.

    Parameters
    ----------
    file_path : str
        Path to the .scool file.
    chrom : str
        Chromosome name.
    diag_num : int
        Number of diagonals to consider.
    balance : bool, optional
        If True, use balanced data. Default is False. This parameter is specific to cooler files.

    Returns
    -------
    Dict[str, band_hic_matrix]
        A dictionary mapping cell IDs to band_hic_matrix objects for the specified chromosome.

    Raises
    ------
    ValueError
        If the scool file is invalid or parameters are incorrect.

    Examples
    --------
    >>> import bandhic as bh
    >>> mats = bh.cooler_chr_all_cells('/Users/wwb/Documents/workspace/BandHiC-Master/data/yeast.10kb.scool', 'chrI', diag_num=10, resolution=10000)
    >>> isinstance(mats['cell1'], band_hic_matrix)
    True
    """
    
    clr = cooler.Cooler(file_path)
    mats = {}
    for cell_id in clr.cell_ids:
        try:
            mats[cell_id] = cooler_chr(
                file_path,
                chrom,
                cell_id=cell_id,
                diag_num=diag_num,
                balance=balance,
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to read cooler group '{file_path}' for chromosome '{chrom}' and cell '{cell_id}': {e}"
            )

# TODO: need test for scool files
def cooler_all_cells_all_chrs(
    file_path: str,
    diag_num: int,
    resolution: Optional[int] = None,
    ) -> Dict[str, Dict[str, band_hic_matrix]]:
    """
    Read Hi-C data from a .scool file for all cells and return a dictionary of dictionaries of band_hic_matrix objects.

    Parameters
    ----------
    file_path : str
        Path to the .scool file.
    diag_num : int
        Number of diagonals to consider.
    resolution : int, optional
        Resolution of the Hi-C data. 

    Returns
    -------
    Dict[str, Dict[str, band_hic_matrix]]
        A dictionary mapping cell IDs to dictionaries mapping chromosome names to band_hic_matrix objects.

    Raises
    ------
    ValueError
        If the scool file is invalid or parameters are incorrect.

    Examples
    --------
    >>> import bandhic as bh
    >>> mats = bh.cooler_all_cells('/Users/wwb/Documents/workspace/BandHiC-Master/data/yeast.10kb.scool', diag_num=10, resolution=10000)
    >>> isinstance(mats['cell1']['chrI'], band_hic_matrix)
    True
    """
    clr = cooler.Cooler(file_path)
    
    mats = {}
    
    for cell_id in clr.cell_ids:
        mats[cell_id] = {}
        for chrom in clr.chromnames:
            mats[cell_id][chrom] = cooler_chr(
                file_path,
                chrom,
                cell_id=cell_id,
                diag_num=diag_num,
                resolution=resolution,
            )
    
    return mats