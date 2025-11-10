from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .utils import make_grid, gaussian_3d, normalize_zero_mean_unit_var


@dataclass
class AttractorSpec:
    """
    Defines an attractor or repeller in the synthetic velocity field.

    Attributes:
        x, y, z : float
            Physical coordinates (in range [0, L)).
        sigma : float
            Width of the Gaussian density bump.
        strength : float
            Strength of the gravitational-like flow.
        kind : str
            Either "attractor" or "repeller".
    """
    x: float
    y: float
    z: float
    sigma: float
    strength: float
    kind: str = "attractor"   # or "repeller"


def _density_contribution(x, y, z, spec: AttractorSpec):
    """
    Gaussian density contribution for one attractor/repeller.
    """
    sign = +1.0 if spec.kind == "attractor" else -1.0
    d = sign * gaussian_3d(x, y, z, spec.x, spec.y, spec.z, spec.sigma)
    return d.astype(np.float32)


def _velocity_contribution(x, y, z, spec: AttractorSpec):
    """
    Physical gravitational-like velocity field:
    
        v(x) ∝ ± (x - x0) / |x - x0|^3

    Attractors: flow inward (minus sign)
    Repellers: flow outward (plus sign)
    """
    # displacement
    dx = x - spec.x
    dy = y - spec.y
    dz = z - spec.z

    # squared radius
    r2 = dx*dx + dy*dy + dz*dz + 1e-6   # avoid division by zero

    # r^3
    r3 = np.sqrt(r2) * r2

    # flow direction
    sign = -1.0 if spec.kind == "attractor" else +1.0

    vx = sign * spec.strength * dx / r3
    vy = sign * spec.strength * dy / r3
    vz = sign * spec.strength * dz / r3

    return (
        vx.astype(np.float32),
        vy.astype(np.float32),
        vz.astype(np.float32),
    )


def generate_fields(
    N: int,
    L: float,
    specs: List[AttractorSpec],
    velocity_scale: float = 1.0,
    density_scale: float = 1.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 3D synthetic fields:
        - density field (Gaussian peaks)
        - velocity field (physical radial flow fields)

    Parameters
    ----------
    N : int
        Grid resolution along each axis.
    L : float
        Physical box size (Mpc/h).
    specs : list of AttractorSpec
        Locations and strengths of attractors/repellers.
    velocity_scale : float
        Multiplier for velocity amplitude.
    density_scale : float
        Multiplier for density amplitude.
    normalize : bool
        If True, normalize fields to zero mean, unit variance.

    Returns
    -------
    d, vx, vy, vz : ndarray
        3D density and velocity fields.
    """

    # Create mesh grid in [0, L)
    x, y, z = make_grid(N, L)

    # Initialize fields
    d  = np.zeros((N, N, N), dtype=np.float32)
    vx = np.zeros_like(d)
    vy = np.zeros_like(d)
    vz = np.zeros_like(d)

    # Summation over all attractors/repellers
    for spec in specs:
        d += density_scale * _density_contribution(x, y, z, spec)

        vxi, vyi, vzi = _velocity_contribution(x, y, z, spec)
        vx += velocity_scale * vxi
        vy += velocity_scale * vyi
        vz += velocity_scale * vzi

    # Optional normalization
    if normalize:
        d  = normalize_zero_mean_unit_var(d)
        vx = normalize_zero_mean_unit_var(vx)
        vy = normalize_zero_mean_unit_var(vy)
        vz = normalize_zero_mean_unit_var(vz)

    return d, vx, vy, vz