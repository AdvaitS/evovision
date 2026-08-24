"""Compare the evolutionary search against random search at a matched budget.

The budget is not chosen in advance: it is however many architectures the
evolutionary search actually trains, which random search is then given. That
removes the usual objection that the baseline was quietly handicapped.

Because this search space is small enough to enumerate (4,096 architectures),
the exact Pareto front is also computable whenever the accuracy function is
cheap, so both searches can be scored as a percentage of ground truth rather
than only against each other.

Usage
-----
    python benchmarks/run_baselines.py --proxy --seeds 5        # seconds
    python benchmarks/run_baselines.py --synthetic --seeds 3    # real training
    python benchmarks/run_baselines.py --subsample 1000         # CIFAR-10
"""

from __future__ import annotations

import argparse
from statistics import median

import numpy as np

from evovision import evolve, search_space
from evovision.baselines import (
    evaluated_budget,
    exhaustive_front,
    front_hypervolume,
    random_search,
)
from evovision.cache import ArchitectureCache


def _proxy_accuracy_fn():
    """A cheap deterministic stand-in: saturating returns to capacity.

    Not a substitute for training, but it has the qualitative shape of an
    accuracy/compute curve and runs in milliseconds, which makes it possible to
    compute the exact front and to run enough seeds for a statistical claim.
    """

    def accuracy_fn(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        out = []
        for x in X:
            cfg = search_space.to_config(x)
            capacity = sum(w * d for w, d in zip(cfg["widths"], cfg["depths"]))
            out.append(0.85 - 0.30 * (1.0 - np.exp(-capacity / 60.0)))
        return np.array(out)

    return accuracy_fn


def _training_accuracy_fn(args):
    from evovision import train_and_eval

    if args.synthetic:
        from evovision.data import synthetic

        train_loader, val_loader = synthetic(n_train=400, n_val=200, seed=args.seed)
    else:
        from evovision.data import cifar10

        train_loader, val_loader = cifar10(subsample=args.subsample, seed=args.seed)

    def accuracy_fn(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.array(
            [train_and_eval(x, train_loader, val_loader, epochs=args.epochs, seed=args.seed) for x in X]
        )

    return lambda: accuracy_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proxy", action="store_true", help="use the cheap analytic proxy")
    parser.add_argument("--synthetic", action="store_true", help="train on the toy dataset")
    args = parser.parse_args()

    make_fn = _proxy_accuracy_fn if args.proxy else _training_accuracy_fn(args)

    print(f"# evovision: evolution vs random search at a matched budget")
    print(f"# search space: {search_space.n_architectures():,} architectures")
    print()

    exact = None
    if args.proxy:
        exact = exhaustive_front(make_fn())
        print(
            f"Exact Pareto front (all {search_space.n_architectures():,} architectures): "
            f"HV={exact['hypervolume']:.5g} over {len(exact['objectives'])} points"
        )
        print()

    print("| seed | architectures trained | evolution HV | random HV | winner |")
    print("|---|---|---|---|---|")
    evo_hv, rand_hv, budgets = [], [], []
    for seed in range(args.seeds):
        result = evolve(make_fn(), pop_size=args.pop_size, generations=args.generations, seed=seed)
        budget = evaluated_budget(result)
        e = front_hypervolume(result.objectives)
        r = random_search(ArchitectureCache(make_fn()), budget, seed=seed)["hypervolume"]
        evo_hv.append(e)
        rand_hv.append(r)
        budgets.append(budget)
        print(
            f"| {seed} | {budget} | {e:.5g} | {r:.5g} | "
            f"{'evolution' if e > r else 'random' if r > e else 'tie'} |"
        )

    print()
    print(f"median budget      : {int(median(budgets))} architectures "
          f"({100 * median(budgets) / search_space.n_architectures():.1f}% of the space)")
    print(f"median evolution HV: {median(evo_hv):.5g}")
    print(f"median random HV   : {median(rand_hv):.5g}")
    wins = sum(e > r for e, r in zip(evo_hv, rand_hv))
    print(f"evolution wins     : {wins}/{args.seeds} seeds")
    if exact is not None:
        print(f"fraction of the exact front reached: "
              f"evolution {100 * median(evo_hv) / exact['hypervolume']:.1f}%, "
              f"random {100 * median(rand_hv) / exact['hypervolume']:.1f}%")
    if args.seeds >= 5:
        try:
            from scipy.stats import wilcoxon

            print(f"Wilcoxon signed-rank p = {wilcoxon(evo_hv, rand_hv).pvalue:.3f}")
        except Exception:  # scipy is optional here
            pass


if __name__ == "__main__":
    main()
