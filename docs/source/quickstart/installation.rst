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
#. joblib >= 1.2
#. numba >= 0.59

There are two recommended ways to install **BandHiC**:

Option 1: Install via pip
--------------------------

If you already have Python >= 3.11 installed:

.. code-block:: bash

   pip install bandhic

If the installation fails due to dependency issues, please manually install the dependencies and then rerun the above command.

Option 2: Install from source with conda
----------------------------------------

1. Clone the repository

   .. code-block:: bash

      git clone https://github.com/xdwwb/BandHiC-Master.git
      cd BandHiC-Master

2. Create the environment and activate it

   .. code-block:: bash

      conda env create -f environment.yml
      conda activate bandhic

3. Install BandHiC

   .. code-block:: bash

      pip install .

Optional dependency for ``.hic`` file support: ``hic-straw``
-----------------------------------------------------------

Support for reading ``.hic`` format Hi-C data relies on the third-party package **hic-straw**, which is **not installed automatically** with BandHiC.

If you do **not** need to read ``.hic`` files, you can ignore this dependency and use BandHiC normally.

If you **do** need ``.hic`` support, please install ``hic-straw`` manually using one of the following methods.

Method 1: Install via pip
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install hic-straw

Note that ``hic-straw`` includes native C/C++ extensions. Installation via ``pip`` may require a compatible compiler toolchain and system libraries (e.g. ``libcurl`` development headers).

Method 2: Install via Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda install -c bioconda hic-straw

Using Conda provides prebuilt binaries on many platforms and avoids local compilation issues.

Upstream installation guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed, system-specific installation instructions, please refer to the official *straw* repository maintained by the Aiden Lab:

https://github.com/aidenlab/straw