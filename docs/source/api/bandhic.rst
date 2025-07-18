.. toctree::
   :maxdepth: 2

`band_hic_matrix` Class API
===========================

.. autoclass:: bandhic.band_hic_matrix
   :members: __init__
   :undoc-members:

Masking Operations
-------------------

These methods allow for the application and manipulation of masks within the matrix. Masks can be used to exclude or highlight specific parts of the matrix during computations.

.. automethod:: bandhic.band_hic_matrix.init_mask
.. automethod:: bandhic.band_hic_matrix.add_mask
.. automethod:: bandhic.band_hic_matrix.add_mask_row_col
.. automethod:: bandhic.band_hic_matrix.get_mask
.. automethod:: bandhic.band_hic_matrix.get_mask_row_col
.. automethod:: bandhic.band_hic_matrix.unmask
.. automethod:: bandhic.band_hic_matrix.clear_mask
.. automethod:: bandhic.band_hic_matrix.drop_mask
.. automethod:: bandhic.band_hic_matrix.drop_mask_row_col
.. automethod:: bandhic.band_hic_matrix.count_masked
.. automethod:: bandhic.band_hic_matrix.count_unmasked
.. automethod:: bandhic.band_hic_matrix.count_in_band_masked
.. automethod:: bandhic.band_hic_matrix.count_out_band_masked
.. automethod:: bandhic.band_hic_matrix.count_in_band
.. automethod:: bandhic.band_hic_matrix.count_out_band


Data Indexing and Modification
-------------------------------

The following methods provide functionality to access, modify, and index the matrix data. These are essential for manipulating individual elements or subsets of the matrix.

.. automethod:: bandhic.band_hic_matrix.__getitem__
.. automethod:: bandhic.band_hic_matrix.__setitem__
.. automethod:: bandhic.band_hic_matrix.get_values
.. automethod:: bandhic.band_hic_matrix.set_values
.. automethod:: bandhic.band_hic_matrix.filled
.. automethod:: bandhic.band_hic_matrix.diag
.. automethod:: bandhic.band_hic_matrix.set_diag




Data Reduction Methods
-----------------------

These methods enable various data reduction operations, such as summing, averaging, or aggregating matrix values along specific axes or dimensions.

.. automethod:: bandhic.band_hic_matrix.min
.. automethod:: bandhic.band_hic_matrix.max
.. automethod:: bandhic.band_hic_matrix.sum
.. automethod:: bandhic.band_hic_matrix.mean
.. automethod:: bandhic.band_hic_matrix.prod
.. automethod:: bandhic.band_hic_matrix.std
.. automethod:: bandhic.band_hic_matrix.var
.. automethod:: bandhic.band_hic_matrix.normalize
.. automethod:: bandhic.band_hic_matrix.ptp
.. automethod:: bandhic.band_hic_matrix.all
.. automethod:: bandhic.band_hic_matrix.any
.. automethod:: bandhic.band_hic_matrix.__contains__


Universal Functions
-------------------
These methods allow for the application of universal functions (ufuncs) to the matrix, enabling element-wise operations similar to those in NumPy.
.. automethod:: numpy.absolute
.. automethod:: numpy.add
.. automethod:: numpy.arccos
.. automethod:: numpy.arccosh
.. automethod:: numpy.arcsin
.. automethod:: numpy.arcsinh
.. automethod:: numpy.arctan
.. automethod:: numpy.arctan2
.. automethod:: numpy.arctanh
.. automethod:: numpy.bitwise_and
.. automethod:: numpy.bitwise_or
.. automethod:: numpy.bitwise_xor
.. automethod:: numpy.cbrt
.. automethod:: numpy.conj
.. automethod:: numpy.conjugate
.. automethod:: numpy.cos
.. automethod:: numpy.cosh
.. automethod:: numpy.deg2rad
.. automethod:: numpy.degrees
.. automethod:: numpy.divide
.. automethod:: numpy.divmod
.. automethod:: numpy.equal
.. automethod:: numpy.exp
.. automethod:: numpy.exp2
.. automethod:: numpy.expm1
.. automethod:: numpy.fabs
.. automethod:: numpy.float_power
.. automethod:: numpy.floor_divide
.. automethod:: numpy.fmod
.. automethod:: numpy.gcd
.. automethod:: numpy.greater
.. automethod:: numpy.greater_equal
.. automethod:: numpy.heaviside
.. automethod:: numpy.hypot
.. automethod:: numpy.invert
.. automethod:: numpy.lcm
.. automethod:: numpy.left_shift
.. automethod:: numpy.less
.. automethod:: numpy.less_equal
.. automethod:: numpy.log
.. automethod:: numpy.log1p
.. automethod:: numpy.log2
.. automethod:: numpy.log10
.. automethod:: numpy.logaddexp
.. automethod:: numpy.logaddexp2
.. automethod:: numpy.logical_and
.. automethod:: numpy.logical_or
.. automethod:: numpy.logical_xor
.. automethod:: numpy.maximum
.. automethod:: numpy.minimum
.. automethod:: numpy.mod
.. automethod:: numpy.multiply
.. automethod:: numpy.negative
.. automethod:: numpy.not_equal
.. automethod:: numpy.positive
.. automethod:: numpy.power
.. automethod:: numpy.rad2deg
.. automethod:: numpy.radians
.. automethod:: numpy.reciprocal
.. automethod:: numpy.remainder
.. automethod:: numpy.right_shift
.. automethod:: numpy.rint
.. automethod:: numpy.sign
.. automethod:: numpy.sin
.. automethod:: numpy.sinh
.. automethod:: numpy.sqrt
.. automethod:: numpy.square
.. automethod:: numpy.subtract
.. automethod:: numpy.tan
.. automethod:: numpy.tanh
.. automethod:: numpy.true_divide


Other Methods
-------------

This section includes additional methods that do not fall into the categories above but provide other useful operations for `bandhic.band_hic_matrix`.

.. automethod:: bandhic.band_hic_matrix.clip
.. automethod:: bandhic.band_hic_matrix.todense
.. automethod:: bandhic.band_hic_matrix.tocoo
.. automethod:: bandhic.band_hic_matrix.tocsr
.. automethod:: bandhic.band_hic_matrix.copy
.. automethod:: bandhic.band_hic_matrix.memory_usage
.. automethod:: bandhic.band_hic_matrix.astype
.. automethod:: bandhic.band_hic_matrix.__repr__
.. automethod:: bandhic.band_hic_matrix.__str__
.. automethod:: bandhic.band_hic_matrix.__len__
.. automethod:: bandhic.band_hic_matrix.__iter__
.. automethod:: bandhic.band_hic_matrix.__hash__
.. automethod:: bandhic.band_hic_matrix.__bool__
.. automethod:: bandhic.band_hic_matrix.__array__
.. automethod:: bandhic.band_hic_matrix.__array_priority__
.. automethod:: bandhic.band_hic_matrix.iterwindows
.. automethod:: bandhic.band_hic_matrix.iterrows
.. automethod:: bandhic.band_hic_matrix.itercols
.. automethod:: bandhic.band_hic_matrix.dump
.. automethod:: bandhic.band_hic_matrix.extract_row


