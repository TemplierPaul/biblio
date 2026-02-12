# JAX Tools - Ecosystem Reference

Quick reference for JAX and its ecosystem libraries for high-performance ML research.

---

## Core JAX

### JAX
**What**: NumPy + autograd + XLA compilation for GPU/TPU
**Key features**:
- `jax.grad`: Automatic differentiation
- `jax.jit`: Just-in-time compilation (XLA)
- `jax.vmap`: Automatic vectorization
- `jax.pmap`: Parallel computation across devices
**Why**: Composable transformations, functional programming, performance
**When to use**: Research code, need speed + flexibility, GPU/TPU acceleration

### JAX JIT Best Practices
**What**: Guidelines for effective use of `@jax.jit` compilation
**Key concepts**:
- Pure functions (no side effects)
- Static vs. traced arguments
- Compilation caching
- Avoiding recompilation pitfalls
**Common issues**: Dynamic shapes, Python control flow, global state
**See**: JAX_JIT_Practices.md and JAX_JIT_Practices_detailed.md

---

## JAX Ecosystem Libraries

### Flax (Neural Networks)
**What**: JAX-based neural network library (similar to PyTorch nn.Module)
**Key components**:
- `flax.linen`: High-level NN API
- `flax.nnx`: New experimental API (simpler state management)
- `flax.training`: Utilities for training loops
**Why**: Functional, explicit state management, composable
**When to use**: Need flexibility beyond high-level APIs, research implementations

### Optax (Optimization)
**What**: Gradient processing and optimization library
**Optimizers**: Adam, SGD, AdamW, RMSProp, Lion, etc.
**Key feature**: Composable gradient transformations (chain multiple transforms)
**Example**: Gradient clipping + weight decay + Adam
**When to use**: Need custom optimizer pipelines, gradient processing

### Flashbax (Experience Replay)
**What**: High-performance replay buffers for RL
**Features**:
- Flat buffers: Simple, fast
- Trajectory buffers: Full episodes
- Prioritized replay: Importance sampling
- JAX-native: Fully JIT-compiled
**When to use**: RL training, need efficient sampling, want JAX integration

### PGX (Game Environments)
**What**: JAX-native game environments (fully differentiable, JIT-compiled)
**Games**: Go, Chess, Shogi, Backgammon, 2048, etc.
**Key advantage**: 1000× faster than Python envs, batch-parallel, TPU-compatible
**When to use**: Game AI research, board games, need massive parallelization

### QDax (Quality-Diversity)
**What**: Quality-Diversity algorithms in JAX (MAP-Elites, CMA-ME, etc.)
**Algorithms**: MAP-Elites, CMA-ME, PGA-ME, QDPG, AURORA
**Features**: Fully JIT-compiled, parallel evaluation, GPU/TPU support
**When to use**: Evolutionary algorithms, QD research, large-scale optimization

### EvoSax (Evolution Strategies)
**What**: Evolution Strategies library (CMA-ES, Sep-CMA-ES, OpenAI ES, etc.)
**Strategies**: CMA-ES, Sep-CMA-ES, Natural ES, OpenAI ES, LM-MA-ES
**Features**: Vectorized across populations, JAX transformations compatible
**When to use**: Black-box optimization, large genomes, parallelizable ES

### Equinox (Neural Networks)
**What**: Alternative to Flax with PyTorch-like interface
**Philosophy**: PyTree-centric, simpler than Flax
**When to use**: Want PyTorch-style API in JAX, less boilerplate than Flax

### Haiku (Neural Networks)
**What**: DeepMind's JAX NN library (predecessor to Flax)
**Note**: Less actively developed, Flax generally preferred now
**When to use**: Legacy DeepMind code, prefer transform-based API

---

## Specialized Tools

### JAXtyping
**What**: Runtime type checking for JAX arrays
**Why**: Catch shape mismatches early, self-documenting code
**Example**: `@jaxtyped` decorator + beartype

### Chex
**What**: Testing and debugging utilities for JAX
**Features**: Shape assertions, fake JIT (debugging), dataclass utilities
**When to use**: Unit tests, debugging JIT issues, development

### Orbax (Checkpointing)
**What**: Efficient checkpointing and model persistence
**Features**: Async checkpointing, large model support, Flax integration
**When to use**: Training long-running models, need robust checkpoints

---

## Summary by Use Case

| Use Case | Tool | Why |
|----------|------|-----|
| Neural networks | Flax | Flexible, functional, research-friendly |
| Optimizers | Optax | Composable, many built-in optimizers |
| RL replay | Flashbax | Fast, JAX-native, prioritized sampling |
| Board games | PGX | 1000× faster, batch-parallel, differentiable |
| Quality-Diversity | QDax | MAP-Elites, GPU/TPU parallel |
| Evolution Strategies | EvoSax | CMA-ES variants, vectorized |
| PyTorch-like API | Equinox | Simpler than Flax, familiar interface |
| Type safety | JAXtyping | Catch shape bugs early |
| Testing | Chex | Debugging JIT, shape assertions |
| Checkpointing | Orbax | Async saves, large models |

---

## Key Advantages of JAX Ecosystem

**Performance**:
- JIT compilation → C++ speed
- Vectorization → Batch parallelism
- Multi-device → GPU/TPU scaling
- XLA optimization → Hardware-specific tuning

**Composability**:
- Functional programming → Easy composition
- Transformations (`jit`, `vmap`, `grad`, `pmap`) → Combine freely
- Pure functions → Reproducible, testable

**Research-Friendly**:
- Low-level control when needed
- High-level abstractions available
- Explicit state management
- Differentiable everything

---

## Common Patterns

### Training Loop
```python
import jax
import flax.linen as nn
import optax

# Model, optimizer, loss
model = nn.Dense(10)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, batch):
    def loss_fn(params):
        logits = model.apply(params, batch['x'])
        return jnp.mean((logits - batch['y'])**2)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss
```

### Vectorization
```python
# Vectorize across batch dimension
batched_forward = jax.vmap(model.apply, in_axes=(None, 0))
outputs = batched_forward(params, inputs)  # Shape: (batch, ...)
```

### Multi-Device
```python
# Parallelize across devices
pmap_forward = jax.pmap(model.apply)
outputs = pmap_forward(params, inputs)  # Distributed computation
```

---

## Relationships

- **JAX** (core) → **Flax/Equinox** (neural networks)
- **JAX** → **Optax** (optimizers)
- **Flax** + **Flashbax** → RL training loops
- **JAX** + **PGX** → Fast game AI
- **JAX** + **QDax** → Quality-Diversity research
- **JAX** + **EvoSax** → Evolution Strategies

---

## Migration Guide

**From PyTorch**:
- `torch.nn.Module` → `flax.linen.Module` or `equinox.Module`
- `torch.optim` → `optax`
- `.backward()` → `jax.grad(loss_fn)`
- `.cuda()` → Automatic device placement with `jax.jit`

**From TensorFlow**:
- `tf.keras.Model` → `flax.linen.Module`
- `tf.GradientTape` → `jax.grad`
- `@tf.function` → `@jax.jit`
- Eager execution → Already default in JAX

---

**For comprehensive JAX ecosystem details, see JAX_Ecosystem_detailed.md**
**For JIT best practices and common pitfalls, see JAX_JIT_Practices_detailed.md**

---

**Official Documentation**:
- JAX: https://jax.readthedocs.io
- Flax: https://flax.readthedocs.io
- Optax: https://optax.readthedocs.io
- QDax: https://qdax.readthedocs.io
