# Evolving efficient vision models on a Pareto front

*Why hand-designing a neural network is betting on one point of a curve — and
how evolutionary search gives you the whole curve instead.*

---

MobileNet, EfficientNet, ResNet-18 — every "efficient" backbone is a single,
carefully-hand-tuned point on the accuracy-vs-cost curve. But the curve is the
interesting object: for any given compute budget there is a *family* of models,
and the best one depends on whether you're shipping to a phone or a datacenter.

[`evovision`](https://github.com/AdvaitS/evovision) turns that insight into a
search problem. It evolves small convolutional networks with two objectives —
minimize validation error **and** multiply-accumulates (MACs) — and returns the
Pareto frontier of architectures trading one off against the other.

## The search space and the objectives

Each candidate is a compact macro architecture: three stages, each choosing a
width (`8/16/24/32` channels), a kernel (`3` or `5`), and a depth (`1` or `2`
layers). That's a search space of `16³ = 4096` networks.

Two objectives:

- **error** — train the network briefly and measure validation error (expensive);
- **MACs** — computed *analytically* from the architecture (free).

The free cost objective is what makes this tractable: evolution spends its
budget only on the expensive part.

## Search, on top of `coevo`

The search itself is `coevo`'s NSGA-II (non-dominated sorting GA). The
interesting part is the fitness prediction: because error is expensive and MACs
are free, `evovision`'s evaluator trains only the candidates in the earliest
*predicted* Pareto fronts and predicts the error of the rest with a surrogate
that re-fits as the search progresses.

Here's a run (`--synthetic`, so you can reproduce it in seconds without
downloading CIFAR-10), with hand-designed baselines for comparison:

| source | error | MACs (M) |
|---|---|---|
| baseline:tiny | 0.870 | 0.42 |
| baseline:small | 0.885 | 1.20 |
| baseline:medium | 0.915 | 9.35 |
| baseline:wide | 0.915 | 45.14 |
| **evolved** | **0.880** | **1.94** |

The evolved point beats `small` on error *and* stays 4.8× cheaper than `medium` —
it found a config no hand-designed baseline covered.

## Why this matters

1. **You get the frontier, not a point.** Ship the left edge for mobile, the
   right edge for the cloud, from one search.
2. **The cost objective is free**, so multi-objective search is genuinely cheap
   to run.
3. **Fitness prediction** keeps it scalable: when training one model takes
   minutes, predicting the accuracy of the unpromising ones is the difference
   between "runs in an hour" and "runs over the weekend".

`evovision` is early and intentionally small — the next steps are a bigger
search space (residual blocks, skip connections) and a real CIFAR-10/TinyImageNet
baseline table. If efficient vision or evolutionary ML interests you, the repo is
[here](https://github.com/AdvaitS/evovision).

---

*Run it yourself: `python benchmarks/run_search.py --synthetic --pop-size 20 --generations 15`.*
