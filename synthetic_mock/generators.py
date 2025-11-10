from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import os

from .fields import AttractorSpec, generate_fields
from .io import save_hdf5, save_npz   # NPZ optional

@dataclass
class RandomConfig:
    num_attractors: int = 1
    num_repellers: int = 0
    sigma_range: tuple = (2.0, 8.0)
    strength_range: tuple = (0.5, 2.0)
    seed: int = 42


def generate_sample(
    N: int,
    L: float,
    cfg: Optional[RandomConfig] = None,
    velocity_scale: float = 1.0,
    density_scale: float = 1.0,
    normalize: bool = True,
    save_dir: Optional[str] = None,
    basename: Optional[str] = None,
    save_npz_file: bool = False,     # NOW OFF BY DEFAULT
    save_hdf5_file: bool = True      # HDF5 is the new default ✅
):
    """
    Generate full synthetic sample and save outputs.
    """
    if cfg is None:
        cfg = RandomConfig()
    rng = np.random.default_rng(cfg.seed)

    # ---- sample attractor/repeller specs ----
    specs = []
    half = L / 2
    for _ in range(cfg.num_attractors):
        x, y, z = rng.uniform(-half, half, size=3)
        sigma = rng.uniform(*cfg.sigma_range)
        strength = rng.uniform(*cfg.strength_range)
        specs.append(AttractorSpec(x, y, z, sigma, strength, "attractor"))

    for _ in range(cfg.num_repellers):
        x, y, z = rng.uniform(-half, half, size=3)
        sigma = rng.uniform(*cfg.sigma_range)
        strength = rng.uniform(*cfg.strength_range)
        specs.append(AttractorSpec(x, y, z, sigma, strength, "repeller"))

    # ---- generate fields ----
    d, vx, vy, vz = generate_fields(
        N, L, specs,
        velocity_scale=velocity_scale,
        density_scale=density_scale,
        normalize=normalize
    )

    # ---- saving paths ----
    if save_dir is None:
        save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    if basename is None:
        basename = f"N{N}_L{int(L)}_A{cfg.num_attractors}_R{cfg.num_repellers}_seed{cfg.seed}"

    h5_path = os.path.join(save_dir, basename + ".hdf5")
    npz_path = os.path.join(save_dir, basename + ".npz")

    # ---- save HDF5 (default) ----
    if save_hdf5_file:
        save_hdf5(h5_path, d, vx, vy, vz)

    # ---- optional NPZ ----
    if save_npz_file:
        save_npz(npz_path, d=d, vx=vx, vy=vy, vz=vz)

    return h5_path, specs