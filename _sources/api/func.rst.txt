Data Reduction and Normalization
--------------------------------
These functions perform data reduction and normalization on BandHiC matrices, allowing for efficient analysis of Hi-C data.
These functions are wrapped around band_hic_matrix methods to provide a consistent interface for data reduction and normalization.

.. autosummary::
   :toctree: ../_autosummary

    bandhic.min
    bandhic.max
    bandhic.sum
    bandhic.mean
    bandhic.std
    bandhic.var
    bandhic.prod
    bandhic.ptp
    bandhic.all
    bandhic.any
    bandhic.clip
    bandhic.normalize

..
    .. autofunction:: bandhic.min
    .. autofunction:: bandhic.max
    .. autofunction:: bandhic.sum
    .. autofunction:: bandhic.mean
    .. autofunction:: bandhic.std
    .. autofunction:: bandhic.var
    .. autofunction:: bandhic.prod
    .. autofunction:: bandhic.ptp
    .. autofunction:: bandhic.all
    .. autofunction:: bandhic.any
    .. autofunction:: bandhic.clip
    .. autofunction:: bandhic.normalize