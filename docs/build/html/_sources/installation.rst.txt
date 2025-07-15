Installation
============

This section describes how to install **BandHiC**.

Requirements
------------

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.6
- cooler
- straw

You can install these using pip:

.. code-block:: bash

   pip install numpy scipy cooler straw

Install from PyPI (if available):

.. code-block:: bash

   pip install bandhic

Or install from source:

.. code-block:: bash

   git clone https://github.com/xdwwb/BandHiC.git
   cd BandHiC
   pip install -e .

To verify the installation:

.. code-block:: python

   import bandHiC
   print(bandHiC.__version__)