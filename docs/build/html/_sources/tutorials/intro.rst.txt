BandHiC Tutorial (Basic)
=========================

Overview
--------

**BandHiC** enables efficient analysis of Hi-C matrices using banded matrix storage and computation.

1. What is a band matrix?
2. Why not use dense / sparse storage?
3. Typical analysis flow:

   - Load data
   - Band conversion
   - Filtering / masking
   - Downstream analysis (e.g., domain detection)

Example: From cooler file
--------------------------

.. code-block:: python

   from bandHiC import cooler_chr
   mat = cooler_chr("data.cool", "chr1", resolution=5000, diag_num=150)

   # Use matrix
   mat.mean(axis="col")


Example: Masking and visualizing
--------------------------------

BandHiC matrices may contain masked regions, such as unmappable genomic bins. You can visualize the masked structure:

.. code-block:: python

   import matplotlib.pyplot as plt
   masked = mat.todense()
   plt.imshow(masked, cmap="Greys", interpolation="none")
   plt.title("Masked BandHiC Matrix")
   plt.show()

Example: Diagonal operations
----------------------------

Because BandHiC is optimized for diagonal band matrices, you can directly access and compute over diagonals:

.. code-block:: python

   main_diag = mat.diag(0)
   offset_diag = mat.diag(3)
   mat.set_diag(0, main_diag * 0.8)

   diag_avg = mat.mean(axis="diag")

Saving and reloading
--------------------

Matrices can be serialized to disk and reloaded:

.. code-block:: python

   mat.dump("demo_matrix.npz")

   from bandHiC import band_hic_matrix
   reloaded = band_hic_matrix.load("demo_matrix.npz")
   print(reloaded.shape)

Window iteration
----------------

You can iterate through sliding windows over the matrix:

.. code-block:: python

   for window in mat.iterwindows(width=5, step=2):
       print(window.shape, window.mean())
Combining multiple matrices
---------------------------

You can combine two BandHiC matrices from the same chromosome using arithmetic operations:

.. code-block:: python

   from bandHiC import read_hic_chr

   mat1 = read_hic_chr("replicate1.hic", method="straw", chrom="chr1", resolution=10000, diag_num=200)
   mat2 = read_hic_chr("replicate2.hic", method="straw", chrom="chr1", resolution=10000, diag_num=200)

   average = (mat1 + mat2) / 2
   diff = mat1 - mat2

Normalization example
---------------------

BandHiC allows simple normalization of diagonal signals:

.. code-block:: python

   diag_mean = mat.mean(axis="diag")
   mat_normalized = (mat - diag_mean) / mat.std(axis="diag")

Batch processing use case
-------------------------

If you process many chromosomes or resolutions, encapsulate logic in functions:

.. code-block:: python

   def load_and_reduce(file, chrom, resolution):
       mat = read_hic_chr(file, method="straw", chrom=chrom, resolution=resolution, diag_num=200)
       return mat.mean(axis="col")

   for chrom in ["chr1", "chr2", "chr3"]:
       avg_col = load_and_reduce("sample.hic", chrom, 10000)
       print(f"{chrom} column average:", avg_col[:5])
Domain detection (basic workflow)
---------------------------------

BandHiC can serve as a lightweight backend for detecting domain boundaries using smoothed diagonal profiles.

Step 1: Extract diagonal band statistics

.. code-block:: python

   profile = mat.mean(axis="diag")

Step 2: Smooth the profile (e.g., moving average)

.. code-block:: python

   import numpy as np
   window_size = 5
   kernel = np.ones(window_size) / window_size
   smoothed = np.convolve(profile, kernel, mode="same")

Step 3: Identify domain boundaries (e.g., local minima)

.. code-block:: python

   from scipy.signal import argrelextrema
   minima = argrelextrema(smoothed, np.less)[0]
   print("Boundary candidates:", minima[:10])

You can further extend this by integrating change point detection or filtering by signal amplitude.


Visualizing domain signal and boundaries
----------------------------------------

You can visualize the smoothed profile and its inferred domain boundaries:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 4))
   plt.plot(smoothed, label="Smoothed Diagonal Signal", color="steelblue")
   plt.scatter(minima, smoothed[minima], color="red", label="Inferred Boundaries")
   plt.xlabel("Diagonal Offset")
   plt.ylabel("Average Contact")
   plt.title("Domain Boundary Detection Profile")
   plt.legend()
   plt.tight_layout()
   plt.show()

Overlaying on heatmap
---------------------

You can also overlay the detected boundaries on the matrix heatmap:

.. code-block:: python

   dense = mat.todense()
   fig, ax = plt.subplots(figsize=(6, 6))
   ax.imshow(dense, cmap="Reds", interpolation="none")
   for d in minima:
       ax.axline((0, d), slope=1, color="blue", linestyle="--", linewidth=1)
   plt.title("Hi-C Matrix with Diagonal Boundaries")
   plt.show()