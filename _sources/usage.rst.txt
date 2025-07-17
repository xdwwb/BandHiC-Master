Usage
=====

This section shows basic usage of the **BandHiC** package.

Read a .hic file and construct a band matrix:

.. code-block:: python

   from bandHiC import read_hic_chr

   mat = read_hic_chr("GM12878.hic", method="straw", chrom="chr1", resolution=10000, diag_num=200)
   print(mat.shape)

Perform a basic matrix operation:

.. code-block:: python

   mat_clipped = mat.clip_copy(0, 100)
   mat_sum = mat.sum(axis="row")

Visualize matrix:

.. code-block:: python

   import matplotlib.pyplot as plt
   plt.imshow(mat.todense(), cmap="Reds")
   plt.colorbar()
   plt.title("BandHiC Matrix")
   plt.show()


Load a .cool file
-----------------

You can also load contact matrices from `.cool` files using the `cooler_chr` function:

.. code-block:: python

   from bandHiC import cooler_chr
   mat = cooler_chr("example.cool", "chr1", resolution=10000, diag_num=150)
   print(mat.shape)

Mathematical operations
-----------------------

BandHiC supports NumPy-like reductions such as `mean`, `sum`, `max`, etc., on rows, columns, or diagonals:

.. code-block:: python

   row_mean = mat.mean(axis="row")
   col_max = mat.max(axis="col")
   diag_sum = mat.sum(axis="diag")

Masking operations
------------------

You can apply or modify the internal mask. To retrieve data while ignoring masked values:

.. code-block:: python

   values = mat.get_values_ignore_mask(from_id, to_id)

Save and reload
---------------

BandHiC matrices can be saved and loaded as compressed `.npz` files:

.. code-block:: python

   mat.dump("matrix.npz")

   from bandHiC import band_hic_matrix
   new_mat = band_hic_matrix.load("matrix.npz")


Advanced matrix operations
--------------------------

BandHiC also supports NumPy-like elementwise operations via overloaded operators:

.. code-block:: python

   mat2 = mat * 2 - 5
   mat3 = mat + mat2

You can also use NumPy ufuncs:

.. code-block:: python

   import numpy as np
   log_mat = np.log1p(mat)
   clipped = np.clip(mat, 0, 100)

Iterating over windows or rows:

.. code-block:: python

   for window in mat.iterwindows(width=10, step=5):
       print(window.shape)

   for row in mat.iterrows():
       print(row.mean())


Matrix combination and manipulation
-----------------------------------

BandHiC matrices can be combined using standard operations:

.. code-block:: python

   combined = (mat + mat2) / 2
   diff = mat - mat2
   normalized = (mat - mat.mean(axis="diag")) / mat.std(axis="diag")

These operations preserve the band structure and mask where applicable.

Command-line usage (optional)
-----------------------------

If you have scripts that use BandHiC and want to apply them in batch mode:

.. code-block:: bash

   python scripts/compute_band_features.py --input GM12878.hic --chrom chr1 --resolution 10000 --diag_num 200

You can integrate BandHiC into workflows using Snakemake, Makefile, or shell scripts.