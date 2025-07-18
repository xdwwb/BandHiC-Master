BandHiC
=====================
Given that most informative chromatin contacts occur within a limited genomic distance (typically within 2 Mb), **BandHiC** adopts a banded storage scheme that stores only a configurable diagonal bandwidth of the dense Hi-C contact matrices. This design can reduce memory usage by up to 99% compared to dense matrices, while still supporting fast random access and user-friendly indexing operations. In addition, BandHiC supports flexible masking mechanisms to efficiently handle missing values, outliers, and unmappable genomic regions. It also provides a suite of vectorized operations optimized with NumPy, making it both scalable and practical for ultra-high-resolution Hi-C data analysis.

.. toctree::
   :maxdepth: 3
   :caption: Contents

   quickstart
   tutorials
   api

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
