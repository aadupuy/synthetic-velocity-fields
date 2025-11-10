#!/usr/bin/env python
import argparse
from synthetic_mock.io import np_to_hdf5

def main():
    p = argparse.ArgumentParser(description="Convert .npz or directory with .npy fields to HDF5.")
    p.add_argument("input", help="Path to .npz file OR directory containing d.npy/vx.npy/vy.npy/vz.npy")
    p.add_argument("output", help="Output .hdf5 path")
    args = p.parse_args()

    np_to_hdf5(args.input, args.output)

if __name__ == "__main__":
    main()