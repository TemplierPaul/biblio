# JAX JIT Practices — Detailed Reference

> **Quick overview**: [[JAX_JIT_Practices]]

## Static Arguments (`static_argnums`)

### When to Use
- Arguments that are **not arrays** (scalars, shapes, dtypes, strings)
- Arguments that **determine control flow** (if statements, loop bounds)
- Arguments that **define shapes** (array dimensions, network architecture)
- Arguments that are **hashable and immutable** (tuples, frozensets)

### Recompilation Problem
```python
@jax.jit(static_argnums=(1,))
def bad(x, n_layers):
    for i in range(n_layers):
        x = layer(x)
    return x

# 100 different compilations!
for n in range(100):
    result = bad(x, n)
```

**Solution**: Bundle static args into a single config:
```python
@dataclass(frozen=True)
class ModelConfig:
    n_layers: int
    hidden_dim: int

@jax.jit(static_argnums=(1,))
def model(x, config: ModelConfig): ...
```

## Dynamic Arguments (Default)

- Array values don't trigger recompilation
- Only shape/dtype changes cause recompilation
- Always prefer dynamic when possible

## Donated Arguments (`donate_argnums`)

### Purpose
Reuse input buffer memory for outputs → fewer allocations.

### When to Use
1. **Replay buffers** (Flashbax requires this)
2. **Large state updates** (optimizer state, model parameters)
3. **Memory-constrained** situations

### Critical Rule
```python
@jax.jit(donate_argnums=(0,))
def update(state, x):
    return state + x

new_state = update(old_state, x)
# ⚠️ old_state is INVALID — never use it again!
```

## Common Patterns

### Conditional Static (training=True/False)
```python
@jax.jit(static_argnums=(2,))
def forward(x, params, training: bool):
    if training:  # Resolved at compile time
        x = dropout(x, 0.1)
    return x
# Exactly 2 compilations: training=T and training=F
```

### Using `static_argnames`
```python
@jax.jit(static_argnames=['training', 'num_classes'])
def model(x, params, training, num_classes): ...
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| JIT inside loop | Define function outside |
| Capturing loop variables | Use `functools.partial` or pass as arg |
| Mutable default arguments | Use `None` default, create inside |
| Non-hashable static args | Use NamedTuple or frozen dataclass |
| Too many static args | Bundle into config object |

## Debugging

```python
# Log recompilations
jax.config.update('jax_log_compiles', True)

# Inspect compiled code
lowered = jax.jit(f).lower(x)
print(lowered.as_text())
```

## PSRO Example

```python
class PSROConfig(NamedTuple):
    population_size: int
    hidden_dim: int
    use_prioritized: bool

@jax.jit(static_argnums=(3,))
def train_best_response(pop_params, meta_strategy, data, config: PSROConfig):
    ...  # config determines shapes; others are dynamic arrays

@jax.jit(donate_argnums=(0,))
def update_replay_buffer(buffer, trajectories):
    return buffer.add(trajectories)
```

## Resources

- [JAX JIT Documentation](https://docs.jax.dev/en/latest/jit-compilation.html)
- [JAX Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)

## Related

- [[JAX_JIT_Practices]] — Quick overview
- [[JAX_Ecosystem_detailed]] — Full ecosystem
- [[JAX Tools]] — Section index
