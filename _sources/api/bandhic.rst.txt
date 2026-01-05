band_hic_matrix Class
=====================

The band_hic_matrix class is the core data structure in the BandHiC package, designed for efficient storage and manipulation of Hi-C contact matrices using a banded representation. It supports various operations similar to those in NumPy, while optimizing memory usage and performance.

.. contents::
   :local:
   :depth: 2

Class Overview and Initialization
---------------------------------

.. currentmodule:: bandhic
.. autoclass:: band_hic_matrix
   :members: __init__

Masking Methods
---------------

These methods allow for the application and manipulation of masks within the matrix. Masks can be used to exclude or highlight specific parts of the matrix during computations.

.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.init_mask
   band_hic_matrix.add_mask
   band_hic_matrix.get_mask
   band_hic_matrix.unmask
   band_hic_matrix.drop_mask
   band_hic_matrix.add_mask_row_col
   band_hic_matrix.get_mask_row_col
   band_hic_matrix.drop_mask_row_col
   band_hic_matrix.clear_mask
   band_hic_matrix.count_masked
   band_hic_matrix.count_unmasked
   band_hic_matrix.count_in_band_masked
   band_hic_matrix.count_out_band_masked
   band_hic_matrix.count_in_band
   band_hic_matrix.count_out_band

..
   .. automethod:: band_hic_matrix.init_mask
   .. automethod:: band_hic_matrix.add_mask
   .. automethod:: band_hic_matrix.add_mask_row_col
   .. automethod:: band_hic_matrix.get_mask
   .. automethod:: band_hic_matrix.get_mask_row_col
   .. automethod:: band_hic_matrix.unmask
   .. automethod:: band_hic_matrix.clear_mask
   .. automethod:: band_hic_matrix.drop_mask
   .. automethod:: band_hic_matrix.drop_mask_row_col
   .. automethod:: band_hic_matrix.count_masked
   .. automethod:: band_hic_matrix.count_unmasked
   .. automethod:: band_hic_matrix.count_in_band_masked
   .. automethod:: band_hic_matrix.count_out_band_masked
   .. automethod:: band_hic_matrix.count_in_band
   .. automethod:: band_hic_matrix.count_out_band

Data Indexing and Modification
------------------------------

The following methods provide functionality to access, modify, and index the matrix data. These are essential for manipulating individual elements or subsets of the matrix.

.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.__getitem__
   band_hic_matrix.__setitem__
   band_hic_matrix.get_values
   band_hic_matrix.set_values
   band_hic_matrix.diag
   band_hic_matrix.set_diag
   band_hic_matrix.extract_row
   band_hic_matrix.__iter__
   band_hic_matrix.iterwindows
   band_hic_matrix.iterrows
   band_hic_matrix.itercols
   band_hic_matrix.filled
   band_hic_matrix.clip

..
   .. automethod:: band_hic_matrix.__getitem__
   .. automethod:: band_hic_matrix.__setitem__
   .. automethod:: band_hic_matrix.get_values
   .. automethod:: band_hic_matrix.set_values
   .. automethod:: band_hic_matrix.filled
   .. automethod:: band_hic_matrix.diag
   .. automethod:: band_hic_matrix.set_diag




Data Reduction and Normalization
--------------------------------

These methods enable various data reduction operations, such as summing, averaging, or aggregating matrix values along specific axes or dimensions.


.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.min
   band_hic_matrix.max
   band_hic_matrix.sum
   band_hic_matrix.mean
   band_hic_matrix.std
   band_hic_matrix.var
   band_hic_matrix.prod
   band_hic_matrix.ptp
   band_hic_matrix.all
   band_hic_matrix.any
   band_hic_matrix.normalize

..
   .. automethod:: band_hic_matrix.min
   .. automethod:: band_hic_matrix.max
   .. automethod:: band_hic_matrix.sum
   .. automethod:: band_hic_matrix.mean
   .. automethod:: band_hic_matrix.std
   .. automethod:: band_hic_matrix.var
   .. automethod:: band_hic_matrix.prod
   .. automethod:: band_hic_matrix.ptp
   .. automethod:: band_hic_matrix.all
   .. automethod:: band_hic_matrix.any
   .. automethod:: band_hic_matrix.clip
   .. automethod:: band_hic_matrix.normalize

Vectorized Computation Methods
-------------------------------
These methods allow for the application of universal functions (ufuncs) to the matrix, enabling element-wise operations similar to those in NumPy.

.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.absolute
   band_hic_matrix.add
   band_hic_matrix.arccos
   band_hic_matrix.arccosh
   band_hic_matrix.arcsin
   band_hic_matrix.arcsinh
   band_hic_matrix.arctan
   band_hic_matrix.arctan2
   band_hic_matrix.arctanh
   band_hic_matrix.bitwise_and
   band_hic_matrix.bitwise_or
   band_hic_matrix.bitwise_xor
   band_hic_matrix.cbrt
   band_hic_matrix.conj
   band_hic_matrix.conjugate
   band_hic_matrix.cos
   band_hic_matrix.cosh
   band_hic_matrix.deg2rad
   band_hic_matrix.degrees
   band_hic_matrix.divide
   band_hic_matrix.divmod
   band_hic_matrix.equal
   band_hic_matrix.exp
   band_hic_matrix.exp2
   band_hic_matrix.expm1
   band_hic_matrix.fabs
   band_hic_matrix.float_power
   band_hic_matrix.floor_divide
   band_hic_matrix.fmod
   band_hic_matrix.gcd
   band_hic_matrix.greater
   band_hic_matrix.greater_equal
   band_hic_matrix.heaviside
   band_hic_matrix.hypot
   band_hic_matrix.invert
   band_hic_matrix.lcm
   band_hic_matrix.left_shift
   band_hic_matrix.less
   band_hic_matrix.less_equal
   band_hic_matrix.log
   band_hic_matrix.log1p
   band_hic_matrix.log2
   band_hic_matrix.log10
   band_hic_matrix.logaddexp
   band_hic_matrix.logaddexp2
   band_hic_matrix.logical_and
   band_hic_matrix.logical_or
   band_hic_matrix.logical_xor
   band_hic_matrix.maximum
   band_hic_matrix.minimum
   band_hic_matrix.mod
   band_hic_matrix.multiply
   band_hic_matrix.negative
   band_hic_matrix.not_equal
   band_hic_matrix.positive
   band_hic_matrix.power
   band_hic_matrix.rad2deg
   band_hic_matrix.radians
   band_hic_matrix.reciprocal
   band_hic_matrix.remainder
   band_hic_matrix.right_shift
   band_hic_matrix.rint
   band_hic_matrix.sign
   band_hic_matrix.sin
   band_hic_matrix.sinh
   band_hic_matrix.sqrt
   band_hic_matrix.square
   band_hic_matrix.subtract
   band_hic_matrix.tan
   band_hic_matrix.tanh
   band_hic_matrix.true_divide

..
   .. automethod:: band_hic_matrix.absolute
   .. automethod:: band_hic_matrix.add
   .. automethod:: band_hic_matrix.arccos
   .. automethod:: band_hic_matrix.arccosh
   .. automethod:: band_hic_matrix.arcsin
   .. automethod:: band_hic_matrix.arcsinh
   .. automethod:: band_hic_matrix.arctan
   .. automethod:: band_hic_matrix.arctan2
   .. automethod:: band_hic_matrix.arctanh
   .. automethod:: band_hic_matrix.bitwise_and
   .. automethod:: band_hic_matrix.bitwise_or
   .. automethod:: band_hic_matrix.bitwise_xor
   .. automethod:: band_hic_matrix.cbrt
   .. automethod:: band_hic_matrix.conj
   .. automethod:: band_hic_matrix.conjugate
   .. automethod:: band_hic_matrix.cos
   .. automethod:: band_hic_matrix.cosh
   .. automethod:: band_hic_matrix.deg2rad
   .. automethod:: band_hic_matrix.degrees
   .. automethod:: band_hic_matrix.divide
   .. automethod:: band_hic_matrix.divmod
   .. automethod:: band_hic_matrix.equal
   .. automethod:: band_hic_matrix.exp
   .. automethod:: band_hic_matrix.exp2
   .. automethod:: band_hic_matrix.expm1
   .. automethod:: band_hic_matrix.fabs
   .. automethod:: band_hic_matrix.float_power
   .. automethod:: band_hic_matrix.floor_divide
   .. automethod:: band_hic_matrix.fmod
   .. automethod:: band_hic_matrix.gcd
   .. automethod:: band_hic_matrix.greater
   .. automethod:: band_hic_matrix.greater_equal
   .. automethod:: band_hic_matrix.heaviside
   .. automethod:: band_hic_matrix.hypot
   .. automethod:: band_hic_matrix.invert
   .. automethod:: band_hic_matrix.lcm
   .. automethod:: band_hic_matrix.left_shift
   .. automethod:: band_hic_matrix.less
   .. automethod:: band_hic_matrix.less_equal
   .. automethod:: band_hic_matrix.log
   .. automethod:: band_hic_matrix.log1p
   .. automethod:: band_hic_matrix.log2
   .. automethod:: band_hic_matrix.log10
   .. automethod:: band_hic_matrix.logaddexp
   .. automethod:: band_hic_matrix.logaddexp2
   .. automethod:: band_hic_matrix.logical_and
   .. automethod:: band_hic_matrix.logical_or
   .. automethod:: band_hic_matrix.logical_xor
   .. automethod:: band_hic_matrix.maximum
   .. automethod:: band_hic_matrix.minimum
   .. automethod:: band_hic_matrix.mod
   .. automethod:: band_hic_matrix.multiply
   .. automethod:: band_hic_matrix.negative
   .. automethod:: band_hic_matrix.not_equal
   .. automethod:: band_hic_matrix.positive
   .. automethod:: band_hic_matrix.power
   .. automethod:: band_hic_matrix.rad2deg
   .. automethod:: band_hic_matrix.radians
   .. automethod:: band_hic_matrix.reciprocal
   .. automethod:: band_hic_matrix.remainder
   .. automethod:: band_hic_matrix.right_shift
   .. automethod:: band_hic_matrix.rint
   .. automethod:: band_hic_matrix.sign
   .. automethod:: band_hic_matrix.sin
   .. automethod:: band_hic_matrix.sinh
   .. automethod:: band_hic_matrix.sqrt
   .. automethod:: band_hic_matrix.square
   .. automethod:: band_hic_matrix.subtract
   .. automethod:: band_hic_matrix.tan
   .. automethod:: band_hic_matrix.tanh
   .. automethod:: band_hic_matrix.true_divide

Data Type Conversion and Representation
---------------------------------------

This section includes methods for converting the matrix to different data types or representations, such as dense, sparse, or masked arrays.

.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.todense
   band_hic_matrix.tocoo
   band_hic_matrix.tocsr
   band_hic_matrix.copy
   band_hic_matrix.astype
   band_hic_matrix.__repr__
   band_hic_matrix.__str__
   band_hic_matrix.__array__
   band_hic_matrix.__array_priority__



Other Methods
-------------

This section includes additional methods that do not fall into the categories above but provide other useful operations for `band_hic_matrix`.

.. autosummary::
   :toctree: ../_autosummary

   band_hic_matrix.memory_usage
   band_hic_matrix.__len__
   band_hic_matrix.__hash__
   band_hic_matrix.__bool__
   band_hic_matrix.dump


..
   .. automethod:: band_hic_matrix.clip
   .. automethod:: band_hic_matrix.todense
   .. automethod:: band_hic_matrix.tocoo
   .. automethod:: band_hic_matrix.tocsr
   .. automethod:: band_hic_matrix.copy
   .. automethod:: band_hic_matrix.memory_usage
   .. automethod:: band_hic_matrix.astype
   .. automethod:: band_hic_matrix.__repr__
   .. automethod:: band_hic_matrix.__str__
   .. automethod:: band_hic_matrix.__len__
   .. automethod:: band_hic_matrix.__iter__
   .. automethod:: band_hic_matrix.__hash__
   .. automethod:: band_hic_matrix.__bool__
   .. automethod:: band_hic_matrix.__array__
   .. automethod:: band_hic_matrix.__array_priority__
   .. automethod:: band_hic_matrix.iterwindows
   .. automethod:: band_hic_matrix.iterrows
   .. automethod:: band_hic_matrix.itercols
   .. automethod:: band_hic_matrix.dump
   .. automethod:: band_hic_matrix.extract_row
