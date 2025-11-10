# synthetic_mock

Generate synthetic 3D **density** and **velocity** fields with attractors and repellers,  
and export them **directly to HDF5** for use in the C++ segmentation pipeline.

---

## 🧬 Scientific Context

This package was developed to support the velocity-based segmentation
method introduced in:

**Partitioning the universe into gravitational basins using the cosmic velocity field**  
*A. Dupuy, H. M. Courtois, F. Dupont, F. Denis, R. Graziani,  
Y. Copin, D. Pomarède, N. Libeskind, E. Carlesi, B. Tully, D. Guinet*  
**MNRAS Letters (2019)**  
DOI: https://doi.org/10.1093/mnrasl/slz115  
arXiv: https://arxiv.org/abs/1907.06555

In that work, the cosmic peculiar velocity field was used to compute **streamlines**
(flow lines). By following each streamline to its convergence point, the local Universe 
can be partitioned into **gravitational basins** (also referred to as velocity-based 
watersheds). This technique provides a natural way to characterize the morphology, 
extent, and boundaries of large-scale structures such as superclusters or repellers.

To rigorously validate the segmentation algorithm, a suite of **synthetic mock
velocity fields** was required. These mocks allow controlled experiments where:

- the number and type of attractors/repellers are known,
- the flow topology is analytically predictable,
- the velocity field is noise-free or noise-controlled,
- the watershed boundaries are expected and reproducible,
- the segmentation code can be stress-tested before being applied to
  real observational data (e.g., *Cosmicflows-4*).

The `synthetic_mock` package provides a clean, extensible, and Pythonic
implementation of these artificial velocity fields. It enables users to generate
customizable 3D test cases for:

- benchmarking gravitational basin segmentation,  
- validating streamline integration approaches,  
- exploring flow topology under controlled conditions,  
- preparing tutorials or teaching materials on cosmic flows.

---

## 📦 Installation (editable mode)

```bash
pip install -e .
```

---

## ✅ Generate one synthetic sample (HDF5 output)

```bash
python scripts/generate_data.py \
    --N 128 \
    --L 160 \
    --attractors 2 \
    --repellers 1 \
    --outdir output
```

This produces: `output/N128_L160_A2_R1_seed42.hdf5`

---

## 🧪 HDF5 Format

Each generated file contains four datasets:
* `d`  — density proxy
* `vx` — x-component of velocity
* `vy` — y-component of velocity
* `vz` — z-component of velocity

All stored as `float32`.

Example structure:

```bash
root
 ├─ d   (N×N×N)
 ├─ vx  (N×N×N)
 ├─ vy  (N×N×N)
 └─ vz  (N×N×N)
 ```

---

## 🐍 Python API

```python
from synthetic_mock import (
    AttractorSpec,
    generate_fields,
    generate_sample,
)
```

Example usage:

```python
from synthetic_mock import AttractorSpec, generate_fields

specs = [
    AttractorSpec(x=80, y=60, z=40, sigma=5, strength=1.2, kind="attractor"),
]

d, vx, vy, vz = generate_fields(128, 160.0, specs)
```

---

## 🧠 Model Notes

* Velocity field follows $$\mathbf{v} \propto -\nabla \phi$$, where $$\phi$$ is a Gaussian-shaped potential well or hill.
* Density proxy uses the same Gaussian with sign depending on attractor/repeller.
* Optional zero-mean / unit-std normalization for
    * `d`,
    * `vx`, `vy`, `vz`.

---

## 📁 Project Structure

```bash
synthetic_mock/
    __init__.py
    fields.py
    generators.py
    io.py
    utils.py
scripts/
    generate_data.py
README.md
```
