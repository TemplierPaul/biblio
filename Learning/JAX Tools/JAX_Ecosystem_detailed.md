# JAX Ecosystem — Detailed Notes

> **Index**: [[JAX Tools]]

## JAX Core

High-performance numerical computing combining NumPy API + autodiff + XLA compilation.

### Composable Transformations
```python
import jax
import jax.numpy as jnp

jax.grad(loss_fn)          # Automatic differentiation
jax.jit(slow_fn)           # JIT compilation
jax.vmap(single_fn)        # Vectorization (batching)
jax.pmap(fn)               # Multi-device parallelization
```

### Why JAX for RL
- 10–1000x faster than NumPy
- Vectorize across population/episodes
- Pure functional → reproducible
- Growing ecosystem (Flax, Optax, RLax, etc.)

---

## Flax NNX

PyTorch-like neural network API for JAX (2024+):

```python
from flax import nnx

class MLP(nnx.Module):
    def __init__(self, in_dim, out_dim, rngs):
        self.linear1 = nnx.Linear(in_dim, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, out_dim, rngs=rngs)

    def __call__(self, x):
        return self.linear2(nnx.relu(self.linear1(x)))
```

**Key features**: Mutable state, automatic parameter collection, `nnx.jit` decorator.

---

## Optax

Gradient processing and optimization:

```python
import optax

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=3e-4)
)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

---

## Flashbax

Hardware-accelerated replay buffers for JAX:

```python
import flashbax as fbx

buffer = fbx.make_flat_buffer(max_length=100000, min_length=1000)
buffer_state = buffer.init(sample_experience)  # Infers structure
buffer_state = buffer.add(buffer_state, new_experience)
batch = buffer.sample(buffer_state, rng_key)
```

**Types**: Flat, Trajectory, Prioritized (PER), Item.

**Critical**: Use `donate_argnums` for memory-efficient buffer updates. See [[JAX_JIT_Practices]].

---

## PGX

GPU-accelerated board game environments:

```python
import pgx

env = pgx.make("go_9x9")
state = env.init(jax.random.PRNGKey(0))
state = env.step(state, action)
# Fully vmap-able → 1000s of games in parallel
```

**Games**: Go, Chess, Shogi, Backgammon, Hex, Connect4, Othello, etc.

---

## QDax

Quality-Diversity in JAX:

```python
import qdax
from qdax.core.map_elites import MAPElites

me = MAPElites(scoring_fn=score, emitter=emitter, metrics_fn=metrics)
repertoire, emitter_state, key = me.init(genotypes, centroids, key)
```

See [[MAP_Elites_detailed]] and [[Quality_Diversity_detailed]].

---

## Code Resources

| Tool | Link |
|------|------|
| JAX | [github.com/jax-ml/jax](https://github.com/jax-ml/jax) |
| Flax NNX | [flax.readthedocs.io](https://flax.readthedocs.io) |
| Optax | [github.com/google-deepmind/optax](https://github.com/google-deepmind/optax) |
| Flashbax | [github.com/instadeepai/flashbax](https://github.com/instadeepai/flashbax) |
| PGX | [github.com/sotetsuk/pgx](https://github.com/sotetsuk/pgx) |
| QDax | [github.com/adaptive-intelligent-robotics/QDax](https://github.com/adaptive-intelligent-robotics/QDax) |

## Related

- [[JAX Tools]] — Section index
- [[JAX_JIT_Practices]] — JIT compilation best practices
- [[RNN_Policies]] — Recurrent policies in JAX
