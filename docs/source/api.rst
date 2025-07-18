API Reference
=============

This page provides a full API overview of the BandHiC module, including the core band_hic_matrix class and its utilities.

The `bandhic` module defines data structures and utilities for efficiently storing and manipulating Hi-C contact matrices in a banded form. It enables NumPy-compatible operations, matrix masking, diagonal reduction, file I/O, and domain-level interaction analysis.

Core components:
- `band_hic_matrix`: A memory-efficient matrix class supporting banded Hi-C data.
- Utility functions: Loaders (`cooler_chr`) and constructors (`ones`, `zeros`, etc.).

.. toctree::
   :maxdepth: 2

   api/bandhic
   api/create
   api/io
   api/utils