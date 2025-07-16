import numpy as np
from .bandhic import band_hic_matrix
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
    band_hic_matrix,
]

# Automatically generate band_hic_matrix methods for common NumPy ufuncs
_UFUNC_DISPATCH = {
    _np_add: "add",
    _np_subtract: "subtract",
    _np_multiply: "multiply",
    # _np_matmul: "matmul",
    _np_divide: "divide",
    _np_logaddexp: "logaddexp",
    _np_logaddexp2: "logaddexp2",
    _np_true_divide: "true_divide",
    _np_floor_divide: "floor_divide",
    _np_negative: "negative",
    _np_positive: "positive",
    _np_power: "power",
    _np_float_power: "float_power",
    _np_remainder: "remainder",
    _np_mod: "mod",
    _np_fmod: "fmod",
    _np_divmod: "divmod",
    _np_absolute: "absolute",
    _np_fabs: "fabs",
    _np_rint: "rint",
    _np_sign: "sign",
    _np_heaviside: "heaviside",
    _np_conj: "conj",
    _np_conjugate: "conjugate",
    _np_exp: "exp",
    _np_exp2: "exp2",
    _np_log: "log",
    _np_log2: "log2",
    _np_log10: "log10",
    _np_expm1: "expm1",
    _np_log1p: "log1p",
    _np_sqrt: "sqrt",
    _np_square: "square",
    _np_cbrt: "cbrt",
    _np_reciprocal: "reciprocal",
    _np_gcd: "gcd",
    _np_lcm: "lcm",
    _np_sin: "sin",
    _np_cos: "cos",
    _np_tan: "tan",
    _np_arcsin: "arcsin",
    _np_arccos: "arccos",
    _np_arctan: "arctan",
    _np_arctan2: "arctan2",
    _np_hypot: "hypot",
    _np_sinh: "sinh",
    _np_cosh: "cosh",
    _np_tanh: "tanh",
    _np_arcsinh: "arcsinh",
    _np_arccosh: "arccosh",
    _np_arctanh: "arctanh",
    _np_degrees: "degrees",
    _np_radians: "radians",
    _np_deg2rad: "deg2rad",
    _np_rad2deg: "rad2deg",
    _np_bitwise_and: "bitwise_and",
    _np_bitwise_or: "bitwise_or",
    _np_bitwise_xor: "bitwise_xor",
    _np_invert: "invert",
    _np_left_shift: "left_shift",
    _np_right_shift: "right_shift",
    _np_greater: "greater",
    _np_greater_equal: "greater_equal",
    _np_less: "less",
    _np_less_equal: "less_equal",
    _np_not_equal: "not_equal",
    _np_equal: "equal",
    _np_logical_and: "logical_and",
    _np_logical_or: "logical_or",
    _np_logical_xor: "logical_xor",
    _np_maximum: "maximum",
    _np_minimum: "minimum",
}

def _generate_ufunc_aliases():
    """
    Dynamically attach NumPy ufunc aliases to band_hic_matrix.
    """
    for ufunc, method_name in _UFUNC_DISPATCH.items():

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
                            f"{uf.__name__} requires an 'other' argument."
                        )
                    other, *rest = args
                    return uf(self, other, *rest, **kwargs)
                # Fallback for ufuncs with more inputs
                else:
                    return uf(self, *args, **kwargs)

            wrapper.__name__ = uf.__name__
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
            wrapper.__doc__ = (
                f"{uf.__name__}({param_sig})\n\n"
                f"Perform element-wise '{uf.__name__}' operation"
                + (" with two inputs." if nin == 2 else ".")
                + "\n\n"
                "Parameters\n"
                "----------\n"
                f"{params_desc}"
                "*args : tuple\n"
                f"    Additional positional arguments for numpy.{uf.__name__}.\n"
                "**kwargs : dict\n"
                f"    Keyword arguments for numpy.{uf.__name__}.\n\n"
                "Returns\n"
                "-------\n"
                "band_hic_matrix\n"
                f"    Result of element-wise '{uf.__name__}' operation.\n\n"
                "See Also\n"
                "--------\n"
                f"numpy.{uf.__name__}\n\n"
                "Examples\n"
                "--------\n"
                + (
                    f">>> from bandhic import band_hic_matrix\n"
                    f">>> mat = band_hic_matrix(np.eye(3), diag_num=2, dtype=int)\n"
                    + (
                        f">>> result = mat.{uf.__name__}()\n"
                        if nin == 1
                        else f">>> other = mat.copy()\n>>> result = mat.{uf.__name__}(other)\n"
                    )
                )
            )
            return wrapper

        setattr(band_hic_matrix, ufunc.__name__, make_ufunc_wrapper(ufunc))

# Execute ufunc alias generation
_generate_ufunc_aliases()