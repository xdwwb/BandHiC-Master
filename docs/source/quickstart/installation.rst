Installation
============

Required Package
-----------------

**BandHiC** can be installed on Linux-like systems and requires the following dependencies:

#. python >= 3.11
#. numpy >= 2.3
#. pandas >= 2.3
#. scipy >= 1.16
#. `cooler >= 0.10 <https://cooler.readthedocs.io/en/latest/>`__
#. `hic_straw >= 1.3 <https://pypi.org/project/hic-straw/>`__

There are two recommended ways to install **BandHiC**:

Option 1: Install via pip
--------------------------

If you already have Python >= 3.11 installed:

.. code-block:: bash

   pip install bandhic

Option 2: Install from source with conda
----------------------------------------

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/xdwwb/BandHiC-Master.git
   cd BandHiC-Master

   # 2. Create the environment and activate it
   conda env create -f environment.yml
   conda activate bandhic

   # 3. Install BandHiC
   pip install .

Build Troubleshooting for hic-straw
------------------------------------

If you encounter an error like the following while installing or building ``hic-straw``:

.. code-block:: text

   fatal error: curl/curl.h: No such file or directory

This means the C++ extension in ``hic-straw`` requires the **libcurl development headers**, which are not installed by default on many systems.

**Solution 1: Install system dependencies (for pip installation)**

You need to install the ``libcurl`` development package before building:

- **On Ubuntu/Debian**:

  .. code-block:: bash

     sudo apt-get update
     sudo apt-get install libcurl4-openssl-dev

- **On Fedora/CentOS/RHEL**:

  .. code-block:: bash

     sudo dnf install libcurl-devel

- **On macOS (with Homebrew)**:

  .. code-block:: bash

     brew install curl

  If Homebrew's curl is not found automatically, you may need to set environment variables:

  .. code-block:: bash

     export CPATH="$(brew --prefix curl)/include"
     export LIBRARY_PATH="$(brew --prefix curl)/lib"

**Solution 2: Use Conda (recommended for convenience)**

Instead of building ``hic-straw`` from source, you can install a prebuilt binary via `Bioconda <https://bioconda.github.io/>`__:

.. code-block:: bash

   conda install -c bioconda hic-straw

To avoid conflicts and ensure reproducibility, we recommend installing it in a fresh Conda environment:

.. code-block:: bash

   conda create -n bandhic-env python=3.11
   conda activate bandhic-env
   conda install -c bioconda hic-straw

   # Install BandHiC
   pip install bandhic