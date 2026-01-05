Tutorials
==========
 
This section provides detailed tutorials on how to use the **BandHiC** package effectively.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

BandHiC can serve as an alternative to the NumPy package when managing and manipulating Hi-C data, aiming to address the issue of excessive memory usage caused by storing dense matrices using NumPy's ``ndarray``. At the same time, BandHiC supports masking operations similar to NumPy’s ``ma.MaskedArray`` module, with enhancements tailored for Hi-C data.

Users can leverage their experience with NumPy when using the BandHiC package, so it is recommended that users have some basic knowledge of NumPy. A link to NumPy is provided below: https://numpy.org

Import ``bandhic`` package
--------------------------

.. code-block:: python

   >>> import bandhic as bh

Initialize a ``band_hic_matrix`` object
---------------------------------------

Initialize from a SciPy ``coo_matrix`` object:

.. code-block:: python

   >>> import bandhic as bh
   >>> import numpy as np
   >>> from scipy.sparse import coo_matrix
   >>> coo = coo_matrix(([1, 2, 3], ([0, 1, 2],[0, 1, 2])), shape=(3,3))
   >>> mat1 = bh.band_hic_matrix(coo, diag_num=2)

Initialize from a tuple (data, (row, col)):

.. code-block:: python

   >>> mat2 = bh.band_hic_matrix(([4, 5, 6], ([0, 1, 2],[2, 1, 0])), diag_num=1)

Initialize from a full dense array, only upper-triangular part is stored, lower part is symmetrized:

.. code-block:: python

   >>> arr = np.arange(16).reshape(4,4)
   >>> mat3 = bh.band_hic_matrix(arr, diag_num=3)

Load or save a ``band_hic_matrix`` object
-----------------------------------------

.. code-block:: python

   >>> bh.save_npz('./sample.npz', mat)
   >>> mat = bh.load_npz('./sample.npz')

Load from ``.hic`` file:

.. code-block:: python

   >>> mat = bh.straw_chr('sample.hic', 
                     'chr1', 
                     resolution=10000, 
                     diag_num=200
                     )

Load from ``.mcool`` file:

.. code-block:: python

   >>> mat = bh.cooler_chr('sample.mcool', 
                     'chr1', 
                     diag_num=200
                     resolution=10000, 
                     )

Construct a ``band_hic_matrix`` object
--------------------------------------

Create a band_hic_matrix object filled with zeros.

.. code-block:: python

   >>> mat1 = bh.zeros((5, 5), diag_num=3, dtype=float)

Create a band_hic_matrix object filled with ones.

.. code-block:: python

   >>> mat2 = bh.ones((5, 5), diag_num=3, dtype=float)

Create a band_hic_matrix object filled as an identity matrix.

.. code-block:: python

   >>> mat3 = bh.eye((5, 5), diag_num=3, dtype=float)

Create a band_hic_matrix object filled with a specified value.

.. code-block:: python

   >>> mat4 = bh.full((5, 5), fill_value=0.1, diag_num=3, dtype=float)

Create a band_hic_matrix object matching another matrix, filled with zeros.

.. code-block:: python

   >>> mat5 = bh.zeros_like(mat1, diag_num=3, dtype=float)

Create a band_hic_matrix object matching another matrix, filled with ones.

.. code-block:: python

   >>> mat6 = bh.ones_like(mat1, diag_num=3, dtype=float)

Create a band_hic_matrix object matching another matrix, filled as an identity matrix.

.. code-block:: python

   >>> mat7 = bh.eye_like(mat1, diag_num=3, dtype=float)


Create a band_hic_matrix object matching another matrix, filled with a specified value.

.. code-block:: python

   >>> mat8 = bh.full_like(mat1, fill_value=0.1, diag_num=3, dtype=float)

Indexing on ``band_hic_matrix``
-------------------------------

First, we create a band_hic_matrix object:

.. code-block:: python

   >>> import numpy as np
   >>> import bandhic as bh
   >>> mat = bh.band_hic_matrix(np.arange(16).reshape(4,4), diag_num=2)

Single-element access (scalar)

.. code-block:: python

   >>> mat[1, 2]
   6

Masked element returns masked

.. code-block:: python

   >>> mat2 = bh.band_hic_matrix(np.eye(4), dtype=int, diag_num=2, mask=([0],[1]))
   >>> mat2[0, 1]
   masked

Square submatrix via two-slice indexing returns band_hic_matrix

.. code-block:: python

   >>> sub = mat[1:3, 1:3]
   >>> isinstance(sub, bh.band_hic_matrix)
   True

Single-axis slice returns band_hic_matrix for square region

.. code-block:: python

   >>> sub2 = mat[0:2]  # equivalent to mat[0:2, 0:2]
   >>> isinstance(sub2, bh.band_hic_matrix)
   True

Fancy indexing returns ndarray or MaskedArray

.. code-block:: python

   >>> arr = mat[[0,2,3], [1,2,0]]
   >>> isinstance(arr, np.ndarray)
   True

   >>> mat.add_mask([0,1],[1,2])  # Add mask to some entries
   >>> masked_arr = mat[[0,1], [1,2]]
   >>> isinstance(masked_arr, np.ma.MaskedArray)
   True

Boolean indexing with band_hic_matrix

.. code-block:: python

   >>> mat3 = bh.band_hic_matrix(np.eye(4), diag_num=2, mask=([0,1],[1,2]))
   >>> bool_mask = mat3 > 0  # Create a boolean mask
   >>> result = mat3[bool_mask]  # Use boolean mask for indexing
   >>> isinstance(result, np.ma.MaskedArray)
   True
   >>> result
   masked_array(data=[1.0, 1.0, 1.0, 1.0],
            mask=[False, False, False, False],
      fill_value=0.0)

Masking
-------

Add item-wise mask:

.. code-block:: python

   >>> mat.add_mask([0, 1], [1, 2])

Add row/column mask:

.. code-block:: python

   >>> mask = np.array([True, False, False])
   >>> mat.add_mask_row_col(mask)

Remove mask for specified indices.

.. code-block:: python

   >>> mat.unmask(( [0],[1] ))

Remove all item-wise mask and row/column mask.

.. code-block:: python

   >>> mat.unmask()

Remove all item-wise mask and row/column mask.

.. code-block:: python

   >>> mat.clear_mask()

Drop all item-wise mask but preserve all row/column mask.

.. code-block:: python

   >>> mat.drop_mask()

Drop all row/column mask.

.. code-block:: python

   >>> mat.drop_mask_row_col()

Access masked `band_hic_matrix` will obtain `np.ma.MaskedArray` object:

.. code-block:: python

   >>> mat.add_mask([0, 1], [1, 2])
   >>> masked_arr = mat[[0,1], [1,2]]
   >>> isinstance(masked_arr, np.ma.MaskedArray)
   True

Universal functions (ufunc)
---------------------------

Universal functions that BandHiC supports:

+------------------+-----------------------------------+------------------+-----------------------------------+
| Function         | Description                       | Function         | Description                       |
+==================+===================================+==================+===================================+
| absolute         | Absolute value                    | add              | Element-wise addition             |
+------------------+-----------------------------------+------------------+-----------------------------------+
| arccos           | Inverse cosine                    | arccosh          | Inverse hyperbolic cosine         |
+------------------+-----------------------------------+------------------+-----------------------------------+
| arcsin           | Inverse sine                      | arcsinh          | Inverse hyperbolic sine           |
+------------------+-----------------------------------+------------------+-----------------------------------+
| arctan           | Inverse tangent                   | arctan2          | Arctangent of y/x with quadrant   |
+------------------+-----------------------------------+------------------+-----------------------------------+
| arctanh          | Inverse hyperbolic tangent        | bitwise_and      | Element-wise bitwise AND          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| bitwise_or       | Element-wise bitwise OR           | bitwise_xor      | Element-wise bitwise XOR          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| cbrt             | Cube root                         | conj             | Complex conjugate                 |
+------------------+-----------------------------------+------------------+-----------------------------------+
| conjugate        | Alias for conj                    | cos              | Cosine function                   |
+------------------+-----------------------------------+------------------+-----------------------------------+
| cosh             | Hyperbolic cosine                 | deg2rad          | Degrees to radians                |
+------------------+-----------------------------------+------------------+-----------------------------------+
| degrees          | Radians to degrees                | divide           | Element-wise division             |
+------------------+-----------------------------------+------------------+-----------------------------------+
| divmod           | Quotient and remainder            | equal            | Element-wise equality test        |
+------------------+-----------------------------------+------------------+-----------------------------------+
| exp              | Exponential                       | exp2             | Base-2 exponential                |
+------------------+-----------------------------------+------------------+-----------------------------------+
| expm1            | exp(x) - 1                        | fabs             | Absolute value (float)            |
+------------------+-----------------------------------+------------------+-----------------------------------+
| float_power      | Floating-point power              | floor_divide     | Integer division (floor)          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| fmod             | Modulo operation                  | gcd              | Greatest common divisor           |
+------------------+-----------------------------------+------------------+-----------------------------------+
| greater          | Element-wise greater-than test    | greater_equal    | Greater-than or equal test        |
+------------------+-----------------------------------+------------------+-----------------------------------+
| heaviside        | Heaviside step function           | hypot            | Euclidean norm                    |
+------------------+-----------------------------------+------------------+-----------------------------------+
| invert           | Bitwise inversion                 | lcm              | Least common multiple             |
+------------------+-----------------------------------+------------------+-----------------------------------+
| left_shift       | Bitwise left shift                | less             | Element-wise less-than test       |
+------------------+-----------------------------------+------------------+-----------------------------------+
| less_equal       | Less-than or equal test           | log              | Natural logarithm                 |
+------------------+-----------------------------------+------------------+-----------------------------------+
| log1p            | log(1 + x)                        | log2             | Base-2 logarithm                  |
+------------------+-----------------------------------+------------------+-----------------------------------+
| log10            | Base-10 logarithm                 | logaddexp        | log(exp(x) + exp(y))              |
+------------------+-----------------------------------+------------------+-----------------------------------+
| logaddexp2       | Base-2 version of logaddexp       | logical_and      | Element-wise logical AND          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| logical_or       | Element-wise logical OR           | logical_xor      | Element-wise logical XOR          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| maximum          | Element-wise maximum              | minimum          | Element-wise minimum              |
+------------------+-----------------------------------+------------------+-----------------------------------+
| mod              | Remainder (modulo)                | multiply         | Element-wise multiplication       |
+------------------+-----------------------------------+------------------+-----------------------------------+
| negative         | Element-wise negation             | not_equal        | Element-wise inequality test      |
+------------------+-----------------------------------+------------------+-----------------------------------+
| positive         | Returns input unchanged           | power            | Raise to power                    |
+------------------+-----------------------------------+------------------+-----------------------------------+
| rad2deg          | Radians to degrees                | radians          | Degrees to radians                |
+------------------+-----------------------------------+------------------+-----------------------------------+
| reciprocal       | Element-wise reciprocal           | remainder        | Modulo remainder                  |
+------------------+-----------------------------------+------------------+-----------------------------------+
| right_shift      | Bitwise right shift               | rint             | Round to nearest integer          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| sign             | Sign of input                     | sin              | Sine function                     |
+------------------+-----------------------------------+------------------+-----------------------------------+
| sinh             | Hyperbolic sine                   | sqrt             | Square root                       |
+------------------+-----------------------------------+------------------+-----------------------------------+
| square           | Square of input                   | subtract         | Element-wise subtraction          |
+------------------+-----------------------------------+------------------+-----------------------------------+
| tan              | Tangent function                  | tanh             | Hyperbolic tangent                |
+------------------+-----------------------------------+------------------+-----------------------------------+
| true_divide      | Division that returns float       |                  |                                   |
+------------------+-----------------------------------+------------------+-----------------------------------+

BandHiC supports these universal functions, and they can be used in the following four ways:

1. As methods of the ``band_hic_matrix`` object:

When two band_hic_matrix objects are involved, their shape and diag_num must match

.. code-block:: python

   >>> mat3 = mat1.add(mat2)
   >>> mat4 = mat1.less(mat2)
   >>> mat5 = mat1.negative()

2. As functions of the ``BandHiC`` package

.. code-block:: python

   >>> mat3 = bh.add(mat1, mat2)
   >>> mat4 = bh.less(mat1, mat2)
   >>> mat5 = bh.negative(mat1)

3. Using mathematical operators:

.. code-block:: python

   >>> mat3 = mat1 + mat2
   >>> mat4 = mat1 < mat2
   >>> mat5 = - mat1

4. Calling NumPy's universal functions:

.. code-block:: python

   >>> mat3 = np.add(mat1, mat2)
   >>> mat4 = np.less(mat1, mat2)
   >>> mat5 = np.negative(mat1)

Other Array Functions
---------------------

+----------+---------------------------------------------------------------+
| Function | Description                                                   |
+==========+===============================================================+
| sum      | Compute the sum of all elements along the specified axis      |
+----------+---------------------------------------------------------------+
| prod     | Compute the product of all elements along the specified axis  |
+----------+---------------------------------------------------------------+
| min      | Return the minimum value along the specified axis             |
+----------+---------------------------------------------------------------+
| max      | Return the maximum value along the specified axis             |
+----------+---------------------------------------------------------------+
| mean     | Compute the arithmetic mean along the specified axis          |
+----------+---------------------------------------------------------------+
| var      | Compute the variance (average squared deviation)              |
+----------+---------------------------------------------------------------+
| std      | Compute the standard deviation (square root of variance)      |
+----------+---------------------------------------------------------------+
| ptp      | Compute the range (max - min) of values along the axis        |
+----------+---------------------------------------------------------------+
| all      | Return ``True`` if all elements evaluate to ``True``          |
+----------+---------------------------------------------------------------+
| any      | Return ``True`` if any element evaluates to ``True``          |
+----------+---------------------------------------------------------------+
| clip     | Limit values to a specified min and max range                 |
+----------+---------------------------------------------------------------+

BandHiC supports these functions, and they can be used in the following three ways:

1. As methods of the ``band_hic_matrix`` object:

Compute the sum of all elements including out-of-band values filled with `default_value`.

.. code-block:: python

   >>> result0 = mat1.sum()

Compute the sum of all elements along the `row` axis

.. code-block:: python

   >>> result1 = mat1.sum(axis=0)
   >>> result1 = mat1.sum(axis='row')

Compute the sum of all elements along the `diag` axis

.. code-block:: python
   
   >>> result2 = mat1.sum(axis='diag')

2. Calling ``BandHiC``'s functions:

.. code-block:: python

   >>> result0 = bh.sum(mat1)
   >>> result1 = bh.sum(mat1, axis=0)
   >>> result2 = bh.sum(mat1, axis='diag')

3. Calling NumPy's functions:

.. code-block:: python

   >>> result0 = np.sum(mat1)
   >>> result1 = np.sum(mat1, axis=0)
   >>> result2 = np.sum(mat1, axis='diag')