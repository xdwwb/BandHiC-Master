What is BandHiC?
================

.. only:: html

    .. figure:: _static/bandhic_illustration.svg
        :width: 100%
        :align: center

        Data structure of BandHiC. 
        Schematic illustration of converting a dense symmetric matrix :math:`A` into a banded 
        representation consisting of a data matrix :math:`D`, an element-wise mask matrix 
        :math:`M`, a row/column mask matrix :math:`X`, and a default value :math:`d` for 
        out-of-band entries. Diagonal elements from :math:`A` are reorganized into columns 
        of :math:`D`; :math:`M` marks missing or outlier entries; :math:`X` indicates masked 
        rows or columns.

.. only:: latex

    .. figure:: _static/bandhic_illustration.pdf
        :width: 100%
        :align: center

        Data structure of BandHiC. 
        Schematic illustration of converting a dense symmetric matrix :math:`A` into a banded 
        representation consisting of a data matrix :math:`D`, an element-wise mask matrix 
        :math:`M`, a row/column mask matrix :math:`X`, and a default value :math:`d` for 
        out-of-band entries. Diagonal elements from :math:`A` are reorganized into columns 
        of :math:`D`; :math:`M` marks missing or outlier entries; :math:`X` indicates masked 
        rows or columns.

Data structure of BandHiC
-------------------------

To address the growing memory demands posed by high-resolution Hi-C data, 
we introduce ``band_hic_matrix``, the core class implemented in the BandHiC package. 
For a Hi-C contact matrix :math:`A \in \mathbb{R}^{n \times n}` at resolution :math:`r`, 
``band_hic_matrix`` retains only the diagonals within a user-defined bandwidth :math:`k`, 
yielding a compact representation :math:`D \in \mathbb{R}^{n \times k}`. 
This format ensures that each column in :math:`D` corresponds to a fixed diagonal of :math:`A`, 
such that the mapping :math:`A[i, j] = D[i, j - i]` holds for :math:`|i - j| \le k`.

The memory efficiency achieved by this strategy is substantial. 
When :math:`k \ll n`, the memory footprint of ``band_hic_matrix`` is reduced from 
:math:`\mathcal{O}(n^2)` to :math:`\mathcal{O}(nk)`. 
For example, assuming a resolution of 1 kb and a bandwidth of 2 Mb (:math:`k = 200`), 
the representation of chromosome 1 of the human genome (~249 Mb) requires 3.7 GB of memory, 
less than 1% of the memory required by the dense matrix (~461.9 GB). 
This compression enables the use of high-resolution Hi-C data on commodity hardware 
without sacrificing random access efficiency.

To further enhance flexibility of usage, ``band_hic_matrix`` integrates an optional two-tier 
masking mechanism. The element-wise mask matrix :math:`M \in \{0,1\}^{n \times k}` allows 
users to selectively ignore missing or outlier contacts, enabling robust statistical estimation 
on unmasked subsets. Additionally, a bin-level mask :math:`X \in \{0,1\}^n` supports the exclusion of entire rows or columns, which is particularly useful for removing repetitive genomic regions devoid of a valid Hi-C signal. These masking features facilitate downstream tasks such as the estimation of average contact intensity at given distances, while confirming statistical validity.

Lastly, a scalar default value :math:`d` is defined to fill in the undefined entries of 
:math:`A` not covered by the band matrix :math:`D`. This default is typically set to 0, 
consistent with the assumption that long-range interactions are negligibly sparse. 
It also ensures that reconstruction of the full matrix :math:`A` (if required) can be 
achieved seamlessly by combining :math:`D`, :math:`M`, :math:`X`, and :math:`d`. Overall, 
``band_hic_matrix`` provides an efficient, flexible data structure for scalable Hi-C data 
analysis.

Functions of BandHiC
--------------------

A key feature of ``band_hic_matrix`` is its direct coordinate mapping between the banded 
matrix *B* and the full dense matrix :math:`A`. For any pair of genomic loci :math:`(i, j)` satisfying 
the band constraint :math:`|i - j| \le k`, the interaction frequency :math:`A[i, j]` 
can be accessed in constant time via :math:`D[i, j - i]`. This structure ensures random 
access in :math:`\mathcal{O}(1)` time, which is critical for performance-sensitive Hi-C 
analyses, particularly when memory constraints prohibit the use of fully dense matrices. 
Data access in ``band_hic_matrix`` is fully consistent with that of a dense matrix, as 
each entry is accessed via :math:`B[i, j] = D[i, j - i] = A[i, j]`, allowing users to 
interact with ``band_hic_matrix`` objects as if they were dense matrices without needing 
to consider the underlying implementation.

Owing to this random-access capability, ``band_hic_matrix`` supports NumPy-like indexing 
semantics, including slicing, boolean indexing, and fancy indexing. This design allows users 
to easily query local chromatin contacts. For instance, a slice operation such as 
:code:`B[i:j, i:j]` retrieves a banded submatrix. Combined with the ``todense`` operation, 
this enables reconstruction of the dense submatrix for downstream analysis or visualization.

In addition to flexible data access, ``band_hic_matrix`` also supports a wide range of 
numerical operations, including element-wise arithmetic operations and reduction operations. 
These operations are implemented using NumPy for efficiency. Standard reduction operations 
such as ``sum``, ``min``, and ``max`` are supported along conventional axes (rows or columns), 
as well as along the diagonal axis, which is particularly useful in summarizing interaction 
intensities by genomic distance. This feature facilitates common Hi-C analyses such as 
distance-decay profiling.

Taken together, ``band_hic_matrix`` combines the memory efficiency of a banded storage 
model with the expressiveness of NumPy's interface. By mimicking both NumPy's ``ndarray`` 
and ``MaskedArray`` behaviors, it provides an intuitive and powerful interface for users, 
substantially lowering the barrier to adoption and promoting integration into existing 
Hi-C data analysis workflows.