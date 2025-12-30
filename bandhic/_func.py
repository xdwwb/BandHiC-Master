# -*- coding: utf-8 -*-
# _func.py

"""
_func: 
Author: Weibing Wang
Date: 2025-06-11
Email: wangweibing@xidian.edu.cn

This module provides functions 
"""

from bandhic import band_hic_matrix
import numpy as np
from typing import Callable, Dict, Optional, Union
from numbers import Number

__all__ = [
    "min",
    "max",
    "sum",
    "mean",
    "std",
    "var",
    "prod",
    "ptp",
    "all",
    "any",
    "clip",
    "normalize",
]


def wrap_band_hic_matrix_methods():
    """
    Dynamically attach band_hic_matrix's internal methods as top-level functions.
    If the method is an instance method (expects 'self' as first argument),
    the wrapper will require the user to pass the instance explicitly as the first argument.
    """
    for attr in __all__:
        method = getattr(band_hic_matrix, attr)
        if callable(method):

            def make_method_wrapper(method, attr):
                def wrapper(*args, **kwargs):
                    """
                    Wrapper function to call `band_hic_matrix` methods.
                    """
                    return method(*args, **kwargs)

                wrapper.__name__ = attr
                doc = method.__doc__ or ""
                wrapper.__doc__ = (
                    f"\tWrapped from `band_hic_matrix.{attr}`\n\n"
                    f"\tRefer to the documentation:\n "
                    f"\t:meth:`bandhic.band_hic_matrix.{attr}`"
                )
                # print(wrapper.__doc__)  # Debugging line to check the docstring)
                return wrapper

            globals()[attr] = make_method_wrapper(method, attr)


wrap_band_hic_matrix_methods()
