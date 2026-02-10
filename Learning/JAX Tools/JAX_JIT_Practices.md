# JAX JIT Best Practices

## Overview
JAX JIT compilation creates specialized code for specific argument types. Understanding static vs dynamic arguments is critical for performance.

## Quick Rules

| Argument Type | JIT Behavior | Use For |
|---------------|-------------|---------|
| **Dynamic** (default) | Traces shape/dtype only | Arrays, parameters |
| **Static** (`static_argnums`) | Triggers recompilation per value | Booleans, shapes, configs |
| **Donated** (`donate_argnums`) | Reuses input memory for output | Large buffers, state updates |

## Essential Patterns

### Config as NamedTuple (static)
```python
class Config(NamedTuple):
    hidden_dim: int
    num_layers: int

@jax.jit(static_argnums=(1,))
def model(x, config: Config):
    ...  # Compiles once per unique config
```

### Buffer Donation (memory-efficient)
```python
@jax.jit(donate_argnums=(0,))
def update_buffer(buffer_state, new_data):
    return buffer_state.update(new_data)
# ⚠️ buffer_state is INVALID after call
```

### Avoid Recompilation
```python
# ❌ BAD: Defining JIT inside loop
for i in range(100):
    @jax.jit
    def f(x): return x + i  # New compilation each time!

# ✅ GOOD: Define once, pass i as argument
@jax.jit
def f(x, i): return x + i
```

## Debugging
```python
jax.config.update('jax_log_compiles', True)  # Log recompilations
```

> Detailed reference: [[JAX_JIT_Practices_detailed]]

## Related
- [[JAX Tools]] — Section index
- [[JAX_Ecosystem_detailed]] — Full ecosystem overview
