# BandHiC

**BandHiC** is a Python package for efficient storage, manipulation, and analysis of Hi-C matrices using a banded matrix representation.

## Overview
Given that most informative chromatin contacts occur within a limited genomic distance (typically within 2 Mb), **BandHiC** adopts a banded storage scheme that stores only a configurable diagonal bandwidth of the dense Hi-C contact matrices. This design can reduce memory usage by up to 99% compared to dense matrices, while still supporting fast random access and user-friendly indexing operations. In addition, BandHiC supports flexible masking mechanisms to efficiently handle missing values, outliers, and unmappable genomic regions. It also provides a suite of vectorized operations optimized with NumPy, making it both scalable and practical for ultra-high-resolution Hi-C data analysis.

---

## 🔧 Installation

### Required Package

**BandHiC** could be installed in a linux-like system and requires the following dependencies. 

    1. python>=3.11
    2. numpy>=2.3
    3. pandas>=2.3
    4. scipy>=1.16
    5. cooler>=0.10
    6. hic_straw>=1.3

There are two recommended ways to install **BandHiC**:

### Option 1: Install via `pip`

If you already have Python ≥ 3.11 installed:

```bash
> pip install bandhic
```

### Option 2: Install from source with `conda`

```bash
# 1. Clone the repository
>>> git clone https://github.com/xdwwb/BandHiC-Master.git
>>> cd BandHiC-Master

# 2. Create the environment and activate it
>>> conda env create -f environment.yml
>>> conda activate bandhic

# 3. Install BandHiC
>>> pip install .
```

---

## 🚀 Quick Start

### Prerequisites

BandHiC can serve as an alternative to the NumPy package when managing and manipulating Hi-C data, aiming to address the issue of excessive memory usage caused by storing dense matrices using NumPy’s `ndarray`. At the same time, BandHiC supports masking operations similar to NumPy’s ma.MaskedArray module, with enhancements tailored for Hi-C data.

Users can leverage their experience with NumPy when using the BandHiC package, so it is recommended that users have some basic knowledge of NumPy. A link to NumPy is provided below.

[https://numpy.org](https://numpy.org)


### Import `bandhic` package
```Python
>>> import bandhic as bh
```

### Initialize a `band_hic_matrix` object
Initialize from a SciPy `coo_matrix` object:
```Python
>>> import bandhic as bh
>>> import numpy as np
>>> from scipy.sparse import coo_matrix
>>> coo = coo_matrix(([1, 2, 3], ([0, 1, 2],[0, 1, 2])), shape=(3,3))
>>> mat1 = bh.band_hic_matrix(coo, diag_num=2)
```

Initialize from a tuple (data, (row, col)):
```Python
>>> mat2 = bh.band_hic_matrix(([4, 5, 6], ([0, 1, 2],[2, 1, 0])), diag_num=1)
```

Initialize from a full dense array, only upper-triangular part is stored, lower part is symmetrized:
```Python
>>> arr = np.arange(16).reshape(4,4)
>>> mat3 = bh.band_hic_matrix(arr, diag_num=3)
```

### Load or save a `band_hic_matrix` object
```Python
>>> bh.save_npz('./sample.npz', mat)
>>> mat = bh.load_npz('./sample.npz')
```
Load from `.hic` file:
```Python
>>> mat = bh.straw_chr('sample.hic', 
                        'chr1', 
                        resolution=10000, 
                        diag_num=200
                        )
```
Load from `.mcool` file:
```Python
>>> mat = bh.cooler_chr('sample.mcool', 
                        'chr1', 
                        diag_num=200
                        resolution=10000, 
                        )
```

### Construct a `band_hic_matrix` object
```Python
# Create a band_hic_matrix object filled with zeros.
>>> mat1 = bh.zeros((5, 5), diag_num=3, dtype=float)

# Create a band_hic_matrix object filled with ones.
>>> mat2 = bh.ones((5, 5), diag_num=3, dtype=float)

# Create a band_hic_matrix object filled as an identity matrix.
>>> mat3 = bh.eye((5, 5), diag_num=3, dtype=float)

# Create a band_hic_matrix object filled with a specified value.
>>> mat4 = bh.full((5, 5), fill_value=0.1, diag_num=3, dtype=float)

# Create a band_hic_matrix object matching another matrix, filled with zeros.
>>> mat5 = bh.zeros_like(mat1, diag_num=3, dtype=float)

# Create a band_hic_matrix object matching another matrix, filled with ones.
>>> mat6 = bh.ones_like(mat1, diag_num=3, dtype=float)

# Create a band_hic_matrix object matching another matrix, filled as an identity matrix.
>>> mat7 = bh.eye_like(mat1, diag_num=3, dtype=float)

# Create a band_hic_matrix object matching another matrix, filled with a specified value.
>>> mat8 = bh.full_like(mat1, fill_value=0.1 diag_num=3, dtype=float)
```
### Indexing on `band_hic_matrix`
```python
# First, we create a band_hic_matrix object:
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
```

### Masking 
```python
# Add item-wise mask:
>>> mat.add_mask([0, 1], [1, 2])

# Add row/column mask:
>>> mask = np.array([True, False, False])
>>> mat.add_mask_row_col(mask)

# Remove mask for specified indices.
>>> mat.unmask(( [0],[1] ))

# Remove all item-wise mask and row/column mask.
>>> mat.unmask()

# Remove all item-wise mask and row/column mask.
>>> mat.clear_mask()

# Drop all item-wise mask but preserve all row/column mask.
>>> mat.drop_mask()

# Drop all row/column mask.
>>> mat.drop_mask_row_col()

# Access masked `band_hic_matrix` will obtain `np.ma.MaskedArray` object:
>>> mat.add_mask([0, 1], [1, 2])
>>> masked_arr = mat[[0,1], [1,2]]
>>> isinstance(masked_arr, np.ma.MaskedArray)
True
```

### Universal functions(`ufunc`)
Universal functions that BandHiC support:
| Function 1       | Description 1                     | Function 2       | Description 2                     |
|------------------|-----------------------------------|------------------|-----------------------------------|
| `absolute`       | Absolute value                    | `add`            | Element-wise addition             |
| `arccos`         | Inverse cosine                    | `arccosh`        | Inverse hyperbolic cosine         |
| `arcsin`         | Inverse sine                      | `arcsinh`        | Inverse hyperbolic sine           |
| `arctan`         | Inverse tangent                   | `arctan2`        | Arctangent of y/x with quadrant   |
| `arctanh`        | Inverse hyperbolic tangent        | `bitwise_and`    | Element-wise bitwise AND          |
| `bitwise_or`     | Element-wise bitwise OR           | `bitwise_xor`    | Element-wise bitwise XOR          |
| `cbrt`           | Cube root                         | `conj`           | Complex conjugate                 |
| `conjugate`      | Alias for `conj`                  | `cos`            | Cosine function                   |
| `cosh`           | Hyperbolic cosine                 | `deg2rad`        | Degrees to radians                |
| `degrees`        | Radians to degrees                | `divide`         | Element-wise division             |
| `divmod`         | Quotient and remainder            | `equal`          | Element-wise equality test        |
| `exp`            | Exponential                       | `exp2`           | Base-2 exponential                |
| `expm1`          | `exp(x) - 1`                      | `fabs`           | Absolute value (float)            |
| `float_power`    | Floating-point power              | `floor_divide`   | Integer division (floor)          |
| `fmod`           | Modulo operation                  | `gcd`            | Greatest common divisor           |
| `greater`        | Element-wise greater-than test    | `greater_equal`  | Greater-than or equal test        |
| `heaviside`      | Heaviside step function           | `hypot`          | Euclidean norm                    |
| `invert`         | Bitwise inversion                 | `lcm`            | Least common multiple             |
| `left_shift`     | Bitwise left shift                | `less`           | Element-wise less-than test       |
| `less_equal`     | Less-than or equal test           | `log`            | Natural logarithm                 |
| `log1p`          | `log(1 + x)`                      | `log2`           | Base-2 logarithm                  |
| `log10`          | Base-10 logarithm                 | `logaddexp`      | `log(exp(x) + exp(y))`            |
| `logaddexp2`     | Base-2 version of logaddexp       | `logical_and`    | Element-wise logical AND          |
| `logical_or`     | Element-wise logical OR           | `logical_xor`    | Element-wise logical XOR          |
| `maximum`        | Element-wise maximum              | `minimum`        | Element-wise minimum              |
| `mod`            | Remainder (modulo)                | `multiply`       | Element-wise multiplication       |
| `negative`       | Element-wise negation             | `not_equal`      | Element-wise inequality test      |
| `positive`       | Returns input unchanged           | `power`          | Raise to power                    |
| `rad2deg`        | Radians to degrees                | `radians`        | Degrees to radians                |
| `reciprocal`     | Element-wise reciprocal           | `remainder`      | Modulo remainder                  |
| `right_shift`    | Bitwise right shift               | `rint`           | Round to nearest integer          |
| `sign`           | Sign of input                     | `sin`            | Sine function                     |
| `sinh`           | Hyperbolic sine                   | `sqrt`           | Square root                       |
| `square`         | Square of input                   | `subtract`       | Element-wise subtraction          |
| `tan`            | Tangent function                  | `tanh`           | Hyperbolic tangent                |
| `true_divide`    | Division that returns float       |                  |                                   |

BandHiC supports these universal functions, and they can be used in the following three ways:
1. As methods of the `band_hic_matrix` object:
```python
# When two band_hic_matrix objects are involved, their shape and diag_num must match
>>> mat3 = mat1.add(mat2)
>>> mat4 = mat1.less(mat2)
>>> mat5 = mat1.negative()
```
2. Using mathematical operators:
```python
>>> mat3 = mat1 + mat2
>>> mat4 = mat1 < mat2
>>> mat5 = - mat1
```

3. Calling NumPy's universal functions:
```python
>>> mat3 = np.add(mat1, mat2)
>>> mat4 = np.less(mat1, mat2)
>>> mat5 = np.negative(mat1)
```

### Other Array Functions
| Function | Description |
|----------|-------------|
| `sum`    | Compute the sum of all elements along the specified axis |
| `prod`   | Compute the product of all elements along the specified axis |
| `min`    | Return the minimum value along the specified axis |
| `max`    | Return the maximum value along the specified axis |
| `mean`   | Compute the arithmetic mean along the specified axis |
| `var`    | Compute the variance (average squared deviation) |
| `std`    | Compute the standard deviation (square root of variance) |
| `ptp`    | Compute the range (max - min) of values along the axis |
| `all`    | Return `True` if all elements evaluate to `True` |
| `any`    | Return `True` if any element evaluates to `True` |
| `clip`   | Limit values to a specified min and max range |

BandHiC supports these functions, and they can be used in the following two ways:
1. As methods of the `band_hic_matrix` object:
```python
# Compute the sum of all elements including out-of-band values filled with `default_value`.

>>> result0 = mat1.sum()

# Compute the sum of all elements along the `row` axis
>>> result1 = mat1.sum(axis=0)
>>> result1 = mat1.sum(axis='row')

# Compute the sum of all elements along the `diag` axis
>>> result2 = mat1.sum(axis='diag')
```

3. Calling NumPy's functions:
```python
# Compute the sum of all elements including out-of-band values filled with `default_value`.
>>> result0 = np.sum(mat1)

# Compute the sum of all elements along the `row` axis
>>> result1 = np.sum(mat1, axis=0)

# Compute the sum of all elements along the `diag` axis
>>> result2 = np.sum(mat1, axis='diag')
```
---

## 📚 Features

- Efficient band matrix structure for Hi-C data
- Seamless NumPy integration (e.g., `sum`, `mean`, `clip`)
- Built-in masking and diagonal access
- Save/load via `.npz`
- Sliding window and row/col iteration
- Supports `.hic` (straw) and `.cool` inputs

---

## 📖 Documentation

For full tutorials and API reference, see the [📄 PDF documentation](./docs/build/latex/bandhic.pdf)

---

## 📝 License

MIT License © 2025 Weibing Wang