import numpy as np

def make_grid(N: int, L: float):
    """Return 3D grid coordinates in box [0, L) with N^3 points."""
    x = np.linspace(0, L, N, endpoint=False, dtype=np.float32)
    y = np.linspace(0, L, N, endpoint=False, dtype=np.float32)
    z = np.linspace(0, L, N, endpoint=False, dtype=np.float32)
    return np.meshgrid(x, y, z, indexing="ij")

def gaussian_3d(x, y, z, x0, y0, z0, sigma):
    """3D isotropic Gaussian centered at (x0,y0,z0)."""
    return np.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / (2.0 * sigma**2))

def grad_gaussian_3d(x, y, z, x0, y0, z0, sigma):
    """Gradient of the 3D Gaussian wrt spatial coords."""
    g = gaussian_3d(x, y, z, x0, y0, z0, sigma)
    coeff = -1.0 / (sigma**2)
    gx = coeff * (x - x0) * g
    gy = coeff * (y - y0) * g
    gz = coeff * (z - z0) * g
    return gx, gy, gz

def normalize_zero_mean_unit_var(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = arr.mean()
    s = arr.std()
    if s < eps:
        return arr * 0.0
    return (arr - m) / s

def seed_everything(seed: int):
    np.random.seed(seed)