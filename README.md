# BandHiC

**BandHiC** is a Python package for efficient storage, manipulation, and analysis of Hi-C matrices using a banded matrix representation.

---

## 🔧 Installation

BandHiC requires Python ≥ 3.8 and the following dependencies:

```bash
pip install numpy scipy cooler straw
```

To install BandHiC from source:

```bash
git clone https://github.com/yourname/BandHiC.git
cd BandHiC
pip install -e .
```

---

## 🚀 Quick Start

```python
from bandHiC import read_hic_chr

mat = read_hic_chr("example.hic", method="straw", chrom="chr1", resolution=10000, diag_num=200)
print(mat.mean(axis="row"))
```

You can also load `.cool` files:

```python
from bandHiC import cooler_chr
mat = cooler_chr("example.cool", "chr1", resolution=10000, diag_num=150)
```

Visualize the matrix:

```python
import matplotlib.pyplot as plt
plt.imshow(mat.todense(), cmap="Reds")
plt.colorbar()
plt.show()
```

---

## 📚 Features

- Efficient band matrix structure for Hi-C data
- Seamless NumPy integration (e.g., `sum`, `mean`, `clip`)
- Built-in masking and diagonal access
- Save/load via `.npz`
- Sliding window and row/col iteration
- Supports `.hic` (straw) and `.cool` inputs

---

## 📖 Documentation

For full tutorials and API reference, see the [📄 PDF documentation](./BandHiC/docs/build/latex/bandhic.pdf)

---

## 🧪 Example Workflows

- Matrix normalization and smoothing
- Domain boundary detection via diagonal profile
- Batch processing across chromosomes
- Visual overlays of masked regions and inferred boundaries

---

## 🤝 Contribution

Feel free to open issues or submit pull requests.  
To contribute:

```bash
git clone https://github.com/xdwwb/BandHiC-Master
cd BandHiC
# install & edit
```

---

## 📝 License

MIT License © 2025 Weibing Wang