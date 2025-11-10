#!/usr/bin/env python
import argparse
from synthetic_mock.generators import generate_sample, RandomConfig

def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic 3D velocity/density fields.")
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--L", type=float, default=160.0)
    p.add_argument("--attractors", type=int, default=1)
    p.add_argument("--repellers", type=int, default=0)
    p.add_argument("--sigma-min", type=float, default=2.0)
    p.add_argument("--sigma-max", type=float, default=8.0)
    p.add_argument("--strength-min", type=float, default=0.5)
    p.add_argument("--strength-max", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--velocity-scale", type=float, default=1.0)
    p.add_argument("--density-scale", type=float, default=1.0)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--outdir", type=str, default="output")
    p.add_argument("--basename", type=str, default=None)
    p.add_argument("--hdf5", action="store_true")
    return p.parse_args()

def main():
    a = parse_args()

    cfg = RandomConfig(
        num_attractors=a.attractors,
        num_repellers=a.repellers,
        sigma_range=(a.sigma_min, a.sigma_max),
        strength_range=(a.strength_min, a.strength_max),
        seed=a.seed,
    )

    h5_path, specs = generate_sample(
        N=a.N, L=a.L, cfg=cfg,
        velocity_scale=a.velocity_scale,
        density_scale=a.density_scale,
        normalize=(not a.no_normalize),
        save_dir=a.outdir,
        basename=a.basename
    )

    print(f"Saved: {h5_path}")
    for i, s in enumerate(specs):
        print(f"  spec[{i}]: kind={s.kind} pos=({s.x:.2f},{s.y:.2f},{s.z:.2f}) "
              f"sigma={s.sigma:.2f} strength={s.strength:.2f}")

if __name__ == "__main__":
    main()