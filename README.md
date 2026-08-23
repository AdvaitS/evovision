# evovision

**Multi-objective evolutionary neural-architecture search for efficient vision.**

`evovision` evolves small convolutional networks that are *simultaneously*
accurate and cheap, producing a **Pareto frontier** of architectures trading off
validation error against compute (MACs). It is built directly on
[`coevo`](https://github.com/AdvaitS/coevo) — `NSGA2` drives the search and its
surrogates predict accuracy so you don't have to train every candidate.

## The idea

Hand-designed backbones (MobileNet, EfficientNet, …) are one point on the
accuracy↔cost curve. Evolutionary NAS samples the *whole curve*:

- **Search space** — a compact macro space: three stages, each with a width
  (`{8,16,24,32}`), kernel (`{3,5}`) and depth (`{1,2}`).
- **Two objectives** — minimize validation error *and* MACs (computed
  analytically, so the cost objective is free).
- **Surrogate-assisted accuracy** — train only the most promising candidates;
  a predictor (from `coevo`) estimates the error of the rest.

## Install

```bash
pip install -e ".[dev]"          # pulls torch, torchvision, numpy, and coevo
```

## Quickstart

```python
from evovision import evolve
from evovision.data import cifar10
from evovision import train_and_eval

train_loader, val_loader = cifar10(subsample=1000)

def accuracy_fn(X):
    return [train_and_eval(x, train_loader, val_loader, epochs=2) for x in X]

result = evolve(accuracy_fn, pop_size=20, generations=15)
print(result.summary())          # Pareto front: N architectures, HV, true evals
```

Or run the CLI:

```bash
python benchmarks/run_search.py --generations 15 --subsample 1000          # exact
python benchmarks/run_search.py --generations 30 --surrogate               # surrogate-assisted
```

## What's inside

| Module | Purpose |
|---|---|
| `search_space` | genome encoding, discrete config rounding, analytic MACs/params |
| `models` | build a `torch.nn.Module` from a genome |
| `training` | short CIFAR-10 training loop, returns validation error |
| `problem` | two-objective `coevo.MultiObjectiveProblem` (error, MACs) |
| `evaluator` | `NASSurrogateEvaluator`: predict error, compute MACs exactly, train only the promising |
| `evolve` | `NSGA2` search returning the Pareto frontier |

## The accuracy predictor

Because error is expensive (a training run) and MACs are free, the surrogate
evaluator trains only the candidates in the earliest *predicted* Pareto fronts.
The predictor is re-fit on the growing archive of trained candidates — the same
coevolved fitness-prediction loop as `coevo`, applied to neural networks.

## Roadmap

- Larger search spaces: residual/inverted-residual blocks, per-block skip connections.
- Latency & parameter-count objectives alongside MACs.
- Multi-fidelity fitness prediction (extrapolate final accuracy from a few epochs).
- CIFAR-100 / TinyImageNet and a comparison table vs. hand-designed backbones.

## License

MIT — see [LICENSE](LICENSE).
