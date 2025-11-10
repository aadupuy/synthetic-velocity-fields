import os
import numpy as np
import h5py
from typing import Any

def save_hdf5(path: str, d: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray):
    """
    Save the four 3D fields (d, vx, vy, vz) directly to an HDF5 file.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("d",  data=d.astype(np.float32))
        f.create_dataset("vx", data=vx.astype(np.float32))
        f.create_dataset("vy", data=vy.astype(np.float32))
        f.create_dataset("vz", data=vz.astype(np.float32))

    print(f"✅ Saved HDF5: {path}")
    return path

def save_npz(path: str, **arrays: Any):
    """Optional helper: save arrays to compressed NPZ."""
    arrays32 = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    np.savez_compressed(path, **arrays32)
    return path