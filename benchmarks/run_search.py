"""Run an evolutionary NAS search on CIFAR-10 and print the Pareto frontier.

Usage
-----
    python benchmarks/run_search.py --generations 15 --subsample 1000
    python benchmarks/run_search.py --surrogate --generations 30
"""

from __future__ import annotations

import argparse

import numpy as np

from evovision import evolve, search_space, train_and_eval
from evovision.data import cifar10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--surrogate", action="store_true", help="use surrogate-assisted accuracy")
    parser.add_argument("--synthetic", action="store_true", help="use a synthetic toy dataset (no download)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.synthetic:
        from evovision.data import synthetic

        train_loader, val_loader = synthetic(n_train=400, n_val=200, seed=args.seed)
    else:
        train_loader, val_loader = cifar10(subsample=args.subsample, seed=args.seed)

    def accuracy_fn(X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                train_and_eval(x, train_loader, val_loader, epochs=args.epochs, seed=args.seed)
                for x in X
            ]
        )

    surrogate = None
    if args.surrogate:
        from coevo import NearestNeighborSurrogate

        surrogate = lambda: NearestNeighborSurrogate()

    result = evolve(
        accuracy_fn,
        pop_size=args.pop_size,
        generations=args.generations,
        surrogate_factory=surrogate,
        seed=args.seed,
    )

    print("# evovision Pareto frontier (error, MACs)")
    print("| error | MACs (M) | params | widths | kernels | depths |")
    print("|---|---|---|---|---|---|")
    for obj, sol in sorted(zip(result.objectives, result.solutions), key=lambda r: r[0][1]):
        cfg = search_space.to_config(sol)
        print(
            f"| {obj[0]:.4f} | {obj[1] / 1e6:.2f} | {search_space.params(sol)} "
            f"| {cfg['widths']} | {cfg['kernels']} | {cfg['depths']} |"
        )


if __name__ == "__main__":
    main()
