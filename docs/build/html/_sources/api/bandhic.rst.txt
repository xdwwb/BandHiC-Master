BandHiC API Reference
=====================

This page provides a full API overview of the BandHiC module, including the core band_hic_matrix class and its utilities.

The `bandhic` module defines data structures and utilities for efficiently storing and manipulating Hi-C contact matrices in a banded form. It enables NumPy-compatible operations, matrix masking, diagonal reduction, file I/O, and domain-level interaction analysis.

Core components:
- `band_hic_matrix`: A memory-efficient matrix class supporting banded Hi-C data.
- Utility functions: Loaders (`read_hic_chr`, `cooler_chr`) and constructors (`ones`, `zeros`, etc.).

Class Summary
-------------

.. autosummary::
   :toctree: ../_autosummary
   :recursive:

   bandhic.band_hic_matrix

Detailed Class Documentation
----------------------------

.. autoclass:: bandhic.band_hic_matrix
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Utility Functions
-----------------

.. autosummary::
   :toctree: ../_autosummary

   bandhic.read_hic_chr
   bandhic.cooler_chr
   bandhic.ones
   bandhic.zeros
   bandhic.eye
   bandhic.full
   bandhic.ones_like
   bandhic.zeros_like
   bandhic.eye_like
   bandhic.full_like
