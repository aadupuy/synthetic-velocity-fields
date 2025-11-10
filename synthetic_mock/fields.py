from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .utils import make_grid, gaussian_3d, grad_gaussian_3d, normalize_zero_mean_unit_var

@dataclass
class AttractorSpec:
    x: float
    y: float
    z: float
    sigma: float
    strength: float
    kind: str = "attractor"  # or "repeller"

def _contribute_feature(x, y, z, spec: AttractorSpec):
    """Return density and velocity contribution for one attractor/repeller."""
    sign = +1.0 if spec.kind == "attractor" else -1.0

    # Potential proxy
    phi = sign * spec.strength * gaussian_3d(
        x, y, z, spec.x, spec.y, spec.z, spec.sigma
    )

    # Velocity ~ -∇phi
    gx, gy, gz = grad_gaussian_3d(
        x, y, z, spec.x, spec.y, spec.z, spec.sigma
    )
    vx = -sign * spec.strength * gx
    vy = -sign * spec.strength * gy
    vz = -sign * spec.strength * gz

    # Density proxy = same Gaussian sign
    d = sign * phi

    return d, vx, vy, vz

def generate_fields(N: int, L: float, specs: List[AttractorSpec],
                    velocity_scale: float = 1.0,
                    density_scale: float = 1.0,
                    normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D fields (d, vx, vy, vz) from attractor/repeller specs.
    """
    x, y, z = make_grid(N, L)

    d  = np.zeros((N, N, N), dtype=np.float32)
    vx = np.zeros_like(d)
    vy = np.zeros_like(d)
    vz = np.zeros_like(d)

    for spec in specs:
        di, vxi, vyi, vzi = _contribute_feature(x, y, z, spec)
        d  += di.astype(np.float32)
        vx += vxi.astype(np.float32)
        vy += vyi.astype(np.float32)
        vz += vzi.astype(np.float32)

    d  *= density_scale
    vx *= velocity_scale
    vy *= velocity_scale
    vz *= velocity_scale

    if normalize:
        d  = normalize_zero_mean_unit_var(d).astype(np.float32)
        vx = normalize_zero_mean_unit_var(vx).astype(np.float32)
        vy = normalize_zero_mean_unit_var(vy).astype(np.float32)
        vz = normalize_zero_mean_unit_var(vz).astype(np.float32)

    return d, vx, vy, vz