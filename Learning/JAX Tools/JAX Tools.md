# JAX Tools for RL & Self-Play

This section covers the JAX ecosystem tools used for implementing RL and self-play algorithms.

## Core

### [[JAX_Core|JAX]]
High-performance numerical computing: autodiff, JIT, vmap, pmap.

### [[Flax_NNX|Flax NNX]]
Neural network API for JAX — PyTorch-like with functional internals.

### [[Optax]]
Gradient processing and optimization (Adam, SGD, schedules, clipping).

## RL & Environments

### [[Flashbax]]
Hardware-accelerated replay buffers — flat, trajectory, prioritized. Uses `donate_argnums` for memory efficiency.

### [[PGX]]
GPU-accelerated board game environments in JAX (Go, Chess, Shogi, Backgammon, etc.).

### [[QDax]]
Quality-Diversity on hardware: MAP-Elites, CMA-ME, PGA-ME in JAX/Brax.

## Practices

### [[JAX_JIT_Practices]]
Static vs dynamic arguments, `donate_argnums`, avoiding recompilation.

---

> These notes consolidate the JAX tools referenced across the vault's RL, Game Theory, and QD content.
