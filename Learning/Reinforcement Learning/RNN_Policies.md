# RNN Policies in JAX

## Overview
Recurrent Neural Network policies for partially observable environments, implemented efficiently in JAX using `lax.scan`.

## Why RNN Policies?
- **Partial observability**: Agent doesn't see full state (POMDP)
- **Temporal dependencies**: Actions depend on history
- **Memory**: Remember past events (poker, strategy games)

## Key JAX Pattern: `lax.scan`

```python
# BAD: Python loop (slow, doesn't JIT)
for x in inputs:
    state = rnn_cell(x, state)

# GOOD: jax.lax.scan (fast, JIT-friendly)
def step(state, x):
    new_state = rnn_cell(x, state)
    return new_state, new_state

final_state, all_states = jax.lax.scan(step, init_state, inputs)
```

## Architecture

```
Observation → Encoder (MLP/Conv) → encoded_obs
                                      ↓
                              RNN Cell (LSTM/GRU)
                            carry = hidden_state
                                      ↓
                              Policy Head → π(a|h)
                              Value Head  → v(h)
```

## Integration
- Works with PPO, SAC, MPO for BR training in PSRO
- Essential for imperfect-information games (poker, Stratego)
- Compatible with NeuPL conditional policies

> Detailed implementation: [[RNN_Policies_detailed]]

## Related
- [[RL_Methods_for_PSRO]] — RL algorithms using these policies
- [[JAX_Tools]] — JAX ecosystem
- [[JAX_JIT_Practices]] — JIT compilation patterns
