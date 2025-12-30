from sys import version
import numpy as np
from bandhic import band_hic_matrix
import bandhic
from typing import Callable, Dict
from numpy import absolute as _np_absolute, ndarray
from numpy import add as _np_add
from numpy import arccos as _np_arccos
from numpy import arccosh as _np_arccosh
from numpy import arcsin as _np_arcsin
from numpy import arcsinh as _np_arcsinh
from numpy import arctan as _np_arctan
from numpy import arctan2 as _np_arctan2
from numpy import arctanh as _np_arctanh
from numpy import bitwise_and as _np_bitwise_and
from numpy import bitwise_or as _np_bitwise_or
from numpy import bitwise_xor as _np_bitwise_xor
from numpy import cbrt as _np_cbrt
from numpy import conj as _np_conj
from numpy import conjugate as _np_conjugate
from numpy import cos as _np_cos
from numpy import cosh as _np_cosh
from numpy import deg2rad as _np_deg2rad
from numpy import degrees as _np_degrees
from numpy import divide as _np_divide
from numpy import divmod as _np_divmod
from numpy import equal as _np_equal
from numpy import exp as _np_exp
from numpy import exp2 as _np_exp2
from numpy import expm1 as _np_expm1
from numpy import fabs as _np_fabs
from numpy import float_power as _np_float_power
from numpy import floor_divide as _np_floor_divide
from numpy import fmod as _np_fmod
from numpy import gcd as _np_gcd
from numpy import greater as _np_greater
from numpy import greater_equal as _np_greater_equal
from numpy import heaviside as _np_heaviside
from numpy import hypot as _np_hypot
from numpy import invert as _np_invert
from numpy import lcm as _np_lcm
from numpy import left_shift as _np_left_shift
from numpy import less as _np_less
from numpy import less_equal as _np_less_equal
from numpy import log as _np_log
from numpy import log1p as _np_log1p
from numpy import log2 as _np_log2
from numpy import log10 as _np_log10
from numpy import logaddexp as _np_logaddexp
from numpy import logaddexp2 as _np_logaddexp2
from numpy import logical_and as _np_logical_and
from numpy import logical_or as _np_logical_or
from numpy import logical_xor as _np_logical_xor
from numpy import maximum as _np_maximum
from numpy import minimum as _np_minimum
from numpy import mod as _np_mod
from numpy import multiply as _np_multiply
from numpy import negative as _np_negative
from numpy import not_equal as _np_not_equal
from numpy import positive as _np_positive
from numpy import power as _np_power
from numpy import rad2deg as _np_rad2deg
from numpy import radians as _np_radians
from numpy import reciprocal as _np_reciprocal
from numpy import remainder as _np_remainder
from numpy import right_shift as _np_right_shift
from numpy import rint as _np_rint
from numpy import sign as _np_sign
from numpy import sin as _np_sin
from numpy import sinh as _np_sinh
from numpy import sqrt as _np_sqrt
from numpy import square as _np_square
from numpy import subtract as _np_subtract
from numpy import tan as _np_tan
from numpy import tanh as _np_tanh
from numpy import true_divide as _np_true_divide

__all__ = [
    "band_hic_matrix",
    "add",
    "subtract",
    "multiply",
    # "matmul",  # Not implemented yet
    "divide",
    "logaddexp",
    "logaddexp2",
    "true_divide",
    "floor_divide",
    "negative",
    "positive",
    "power",
    "float_power",
    "remainder",
    "mod",
    "fmod",
    "divmod",
    "absolute",
    "fabs",
    "rint",
    "sign",
    "heaviside",
    "conj",
    "conjugate",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    "gcd",
    "lcm",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "degrees",
    "radians",
    "deg2rad",
    "rad2deg",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
]

# Automatically generate band_hic_matrix methods for common NumPy ufuncs
_UFUNC_DISPATCH = {
    "add": _np_add,
    "subtract": _np_subtract,
    "multiply": _np_multiply,
    "divide": _np_divide,
    "logaddexp": _np_logaddexp,
    "logaddexp2": _np_logaddexp2,
    "true_divide": _np_true_divide,
    "floor_divide": _np_floor_divide,
    "negative": _np_negative,
    "positive": _np_positive,
    "power": _np_power,
    "float_power": _np_float_power,
    "remainder": _np_remainder,
    "mod": _np_mod,
    "fmod": _np_fmod,
    "divmod": _np_divmod,
    "absolute": _np_absolute,
    "fabs": _np_fabs,
    "rint": _np_rint,
    "sign": _np_sign,
    "heaviside": _np_heaviside,
    "conj": _np_conj,
    "conjugate": _np_conjugate,
    "exp": _np_exp,
    "exp2": _np_exp2,
    "log": _np_log,
    "log2": _np_log2,
    "log10": _np_log10,
    "expm1": _np_expm1,
    "log1p": _np_log1p,
    "sqrt": _np_sqrt,
    "square": _np_square,
    "cbrt": _np_cbrt,
    "reciprocal": _np_reciprocal,
    "gcd": _np_gcd,
    "lcm": _np_lcm,
    "sin": _np_sin,
    "cos": _np_cos,
    "tan": _np_tan,
    "arcsin": _np_arcsin,
    "arccos": _np_arccos,
    "arctan": _np_arctan,
    "arctan2": _np_arctan2,
    "hypot": _np_hypot,
    "sinh": _np_sinh,
    "cosh": _np_cosh,
    "tanh": _np_tanh,
    "arcsinh": _np_arcsinh,
    "arccosh": _np_arccosh,
    "arctanh": _np_arctanh,
    "degrees": _np_degrees,
    "radians": _np_radians,
    "deg2rad": _np_deg2rad,
    "rad2deg": _np_rad2deg,
    "bitwise_and": _np_bitwise_and,
    "bitwise_or": _np_bitwise_or,
    "bitwise_xor": _np_bitwise_xor,
    "invert": _np_invert,
    "left_shift": _np_left_shift,
    "right_shift": _np_right_shift,
    "greater": _np_greater,
    "greater_equal": _np_greater_equal,
    "less": _np_less,
    "less_equal": _np_less_equal,
    "not_equal": _np_not_equal,
    "equal": _np_equal,
    "logical_and": _np_logical_and,
    "logical_or": _np_logical_or,
    "logical_xor": _np_logical_xor,
    "maximum": _np_maximum,
    "minimum": _np_minimum,
}


def _generate_ufunc_aliases():
    """
    Dynamically attach NumPy ufunc aliases to band_hic_matrix.
    """
    for method_name, ufunc in _UFUNC_DISPATCH.items():

        def make_ufunc_wrapper(uf):
            nin = uf.nin

            def wrapper(self, *args, **kwargs):
                # Handle unary ufunc (nin=1)
                if nin == 1:
                    return uf(self, *args, **kwargs)
                # Handle binary ufunc (nin=2)
                elif nin == 2:
                    if len(args) < 1:
                        raise TypeError(
                            f"{method_name} requires an 'other' argument."
                        )
                    other, *rest = args
                    return uf(self, other, *rest, **kwargs)
                # Fallback for ufuncs with more inputs
                else:
                    return uf(self, *args, **kwargs)

            wrapper.__name__ = method_name
            # Build a NumPy-style docstring reflecting input count
            if nin == 1:
                param_sig = "self, *args, **kwargs"
                params_desc = "self : band_hic_matrix\n" "    Input matrix.\n"
            else:
                param_sig = "self, other, *args, **kwargs"
                params_desc = (
                    "self : band_hic_matrix\n"
                    "    First input matrix.\n"
                    "other : band_hic_matrix or array-like\n"
                    "    Second input for the operation.\n"
                )
            url = _make_numpy_ufunc_doc_url(method_name)
            wrapper.__doc__ = (
                f"{method_name}({param_sig})\n\n"
                f"Perform element-wise '{method_name}' operation"
                + (" with two inputs." if nin == 2 else ".")
                + "\n\n"
                "Parameters\n"
                "----------\n"
                f"{params_desc}"
                "*args : tuple\n"
                f"    Additional positional arguments for numpy.{method_name}.\n"
                "**kwargs : dict\n"
                f"    Keyword arguments for numpy.{method_name}.\n\n"
                "Returns\n"
                "-------\n"
                "band_hic_matrix\n"
                f"    Result of element-wise '{method_name}' operation.\n\n"
                "See Also\n"
                "--------\n"
                f"`numpy.{method_name} <{url}>`_\n\n"
                "Examples\n"
                "--------\n"
                + (
                    f">>> from bandhic import band_hic_matrix\n"
                    f">>> mat = band_hic_matrix(np.eye(3), diag_num=2, dtype=int)\n"
                    + (
                        f">>> result = mat.{method_name}()\n"
                        if nin == 1
                        else f">>> other = mat.copy()\n>>> result = mat.{method_name}(other)\n"
                    )
                )
            )
            return wrapper

        setattr(band_hic_matrix, method_name, make_ufunc_wrapper(ufunc))


def _generate_ufunc_module_aliases():
    """
    Dynamically attach NumPy ufunc aliases to bandhic module.
    """
    for method_name, ufunc in _UFUNC_DISPATCH.items():

        def make_ufunc_wrapper(uf):
            nin = uf.nin

            def wrapper(*args, **kwargs):
                # Handle unary ufunc (nin=1)
                # Build a NumPy-style docstring reflecting input count
                return uf(*args, **kwargs)

            if nin == 1:
                param_sig = "x, *args, **kwargs"
                params_desc = "x : band_hic_matrix\n" "    Input matrix.\n"
            else:
                param_sig = "x1, x2, *args, **kwargs"
                params_desc = (
                    "x1 : band_hic_matrix, ndarray or scalar\n"
                    "    First input matrix.\n"
                    "x2 : band_hic_matrix, ndarray or scalar\n"
                    "    Second input for the operation.\n"
                )
            url = _make_numpy_ufunc_doc_url(method_name)
            wrapper.__doc__ = (
                f"{method_name}({param_sig})\n\n"
                f"Perform element-wise '{method_name}' operation"
                + (" with two inputs." if nin == 2 else ".")
                + "\n\n"
                "Parameters\n"
                "----------\n"
                f"{params_desc}"
                "*args : tuple\n"
                f"    Additional positional arguments for numpy.{method_name}.\n"
                "**kwargs : dict\n"
                f"    Keyword arguments for numpy.{method_name}.\n\n"
                "Returns\n"
                "-------\n"
                "band_hic_matrix\n"
                f"    Result of element-wise '{method_name}' operation.\n\n"
                "See Also\n"
                "--------\n"
                f"`numpy.{method_name} <{url}>`_\n\n"
                "Examples\n"
                "--------\n"
                + (
                    f">>> import bandhic as bh\n"
                    f">>> x1 = band_hic_matrix(np.eye(3), diag_num=2, dtype=int)\n"
                    + (
                        f">>> result = bh.{method_name}(x1)\n"
                        if nin == 1
                        else f">>> x2 = x1.copy()\n>>> result = bh.{method_name}(x1, x2)\n"
                    )
                )
            )
            wrapper.__name__ = method_name
            wrapper.__module__ = bandhic.__name__
            return wrapper

        # Attach to bandhic module
        func = make_ufunc_wrapper(ufunc)
        globals()[method_name] = func


def _make_numpy_ufunc_doc_url(func_name: str) -> str:
    """
    Construct the NumPy documentation URL for a given ufunc.

    Parameters
    ----------
    func_name : str
        The name of the function, e.g., "add" or "multiply".

    Returns
    -------
    url : str
        The full URL to the function's documentation.
    """
    version = ".".join(np.__version__.split(".")[:2])
    return f"https://numpy.org/doc/{version}/reference/generated/numpy.{func_name}.html"


# Execute ufunc alias generation
_generate_ufunc_aliases()
_generate_ufunc_module_aliases()
