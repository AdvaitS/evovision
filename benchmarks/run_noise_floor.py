"""Measure how much of an accuracy measurement is signal and how much is the seed.

A short training run is a noisy estimate of an architecture's quality. If that
noise is comparable to the spread between architectures, a single-seed search
ranks seeds rather than designs, and no conclusion drawn from it survives.
Run this before trusting any search result from this repo.

Usage
-----
    python benchmarks/run_noise_floor.py --synthetic
    python benchmarks/run_noise_floor.py --subsample 1000 --repeats 5
"""

from __future__ import annotations

import argparse

import numpy as np

from evovision import search_space, train_and_eval
from evovision.noise import measure_noise_floor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5, help="training seeds per architecture")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--synthetic", action="store_true", help="use the toy dataset")
    args = parser.parse_args()

    if args.synthetic:
        from evovision.data import synthetic

        train_loader, val_loader = synthetic(n_train=400, n_val=300, seed=0)
        label = "synthetic gratings"
    else:
        from evovision.data import cifar10

        train_loader, val_loader = cifar10(subsample=args.subsample, seed=0)
        label = f"CIFAR-10 (subsample={args.subsample})"

    def accuracy_fn(genome, seed):
        return train_and_eval(genome, train_loader, val_loader, epochs=args.epochs, seed=seed)

    names = list(search_space.baseline_genomes())
    floor = measure_noise_floor(accuracy_fn, repeats=args.repeats)

    print(f"# evovision accuracy-proxy noise floor — {label}, {args.epochs} epochs")
    print()
    print("| architecture | MACs (M) | mean error | seed sd |")
    print("|---|---|---|---|")
    for name, mean, sd in zip(names, floor.per_architecture_mean, floor.per_architecture_sd):
        genome = search_space.baseline_genomes()[name]
        print(f"| {name} | {search_space.flops(genome) / 1e6:.2f} | {mean:.4f} | {sd:.4f} |")

    print()
    print(f"architecture signal (sd across architectures) : {floor.signal:.4f}")
    print(f"seed noise (mean within-architecture sd)      : {floor.noise:.4f}")
    print(f"signal-to-noise ratio                         : {floor.snr:.2f}")
    print(f"repeats needed for SNR 3                      : {floor.repeats_for_snr(3.0)}")
    print()
    print(floor.summary())
    if floor.snr < 1.0:
        print()
        print("WARNING: noise exceeds signal. A single-seed search over this proxy is")
        print("ranking training seeds, not architectures. Raise `repeats`, train longer,")
        print("or use more data before drawing conclusions from any search result.")


if __name__ == "__main__":
    main()
