"""
Synthetic mock data generation for 3D velocity fields with attractors/repellers.
"""

from .fields import generate_fields, AttractorSpec
from .io import save_hdf5, save_npz   # save_npz optional
from .generators import generate_sample