"""Minimal evovision example with a cheap proxy accuracy function.

Run with::

    python examples/quickstart.py
"""

import numpy as np

from evovision import evolve, to_config
from evovision.search_space import WIDTHS


def proxy_accuracy(X: np.ndarray) -> np.ndarray:
    """Cheap stand-in for training: wider networks get lower error."""
    out = []
    for x in X:
        cfg = to_config(x)
        frac = np.mean([WIDTHS.index(w) / 3.0 for w in cfg["widths"]])
        out.append(0.45 - 0.30 * frac)
    return np.array(out)


def main() -> None:
    result = evolve(proxy_accuracy, pop_size=30, generations=30, seed=0)
    print("Pareto frontier (error, MACs):")
    for obj, sol in sorted(zip(result.objectives, result.solutions), key=lambda r: r[0][1]):
        print(f"  error={obj[0]:.4f}  MACs={obj[1]/1e6:.2f}M  {to_config(sol)['widths']}")


if __name__ == "__main__":
    main()
