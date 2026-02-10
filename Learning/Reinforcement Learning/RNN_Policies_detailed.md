# RNN Policies — Detailed Implementation Notes

> **Quick overview**: [[RNN_Policies]]

## Source

Based on practical JAX implementation patterns for recurrent policies in RL.

## LSTM Cell Implementation (Flax NNX)

```python
import jax
import jax.numpy as jnp
from flax import nnx

class LSTMCell(nnx.Module):
    def __init__(self, input_dim, hidden_dim, rngs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_ii = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.W_hi = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.W_if = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.W_hf = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.W_ig = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.W_hg = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)
        self.W_io = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.W_ho = nnx.Linear(hidden_dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, x, carry):
        h, c = carry
        i = jax.nn.sigmoid(self.W_ii(x) + self.W_hi(h))
        f = jax.nn.sigmoid(self.W_if(x) + self.W_hf(h))
        g = jnp.tanh(self.W_ig(x) + self.W_hg(h))
        o = jax.nn.sigmoid(self.W_io(x) + self.W_ho(h))
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new

    def init_carry(self, batch_size=None):
        shape = (self.hidden_dim,) if batch_size is None else (batch_size, self.hidden_dim)
        return (jnp.zeros(shape), jnp.zeros(shape))
```

## Recurrent Policy Network

```python
class RecurrentPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, rngs=None):
        self.encoder = nnx.Linear(obs_dim, hidden_dim, rngs=rngs)
        self.rnn = LSTMCell(hidden_dim, hidden_dim, rngs=rngs)
        self.policy_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.hidden_dim = hidden_dim

    def __call__(self, obs, carry):
        x = jax.nn.relu(self.encoder(obs))
        new_carry, h = self.rnn(x, carry)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value, new_carry

    def init_carry(self, batch_size=None):
        return self.rnn.init_carry(batch_size)
```

## Efficient Sequence Processing with `lax.scan`

```python
def process_sequence(policy, observations, initial_carry):
    """Process a full trajectory efficiently."""
    def step(carry, obs):
        logits, value, new_carry = policy(obs, carry)
        return new_carry, (logits, value)

    final_carry, (all_logits, all_values) = jax.lax.scan(
        step, initial_carry, observations
    )
    return all_logits, all_values, final_carry
```

## Batched Inference with `vmap`

```python
# Process B trajectories × T timesteps efficiently
batched_process = jax.vmap(process_sequence, in_axes=(None, 0, 0))
# observations: (B, T, obs_dim) → all_logits: (B, T, action_dim)
```

## GRU Alternative

```python
class GRUCell(nnx.Module):
    def __init__(self, input_dim, hidden_dim, rngs):
        self.W_z = nnx.Linear(input_dim + hidden_dim, hidden_dim, rngs=rngs)
        self.W_r = nnx.Linear(input_dim + hidden_dim, hidden_dim, rngs=rngs)
        self.W_h = nnx.Linear(input_dim + hidden_dim, hidden_dim, rngs=rngs)

    def __call__(self, x, carry):
        h = carry
        xh = jnp.concatenate([x, h], axis=-1)
        z = jax.nn.sigmoid(self.W_z(xh))   # Update gate
        r = jax.nn.sigmoid(self.W_r(xh))   # Reset gate
        h_tilde = jnp.tanh(self.W_h(jnp.concatenate([x, r * h], axis=-1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new, h_new
```

## Training with PPO (Recurrent)

```python
@jax.jit
def ppo_recurrent_update(policy, trajectories, old_log_probs, advantages, carry):
    def loss_fn(params):
        logits, values, _ = process_sequence(policy, trajectories.obs, carry)
        log_probs = log_softmax(logits).gather(trajectories.actions)

        ratio = jnp.exp(log_probs - old_log_probs)
        clipped = jnp.clip(ratio, 1-ε, 1+ε) * advantages
        policy_loss = -jnp.minimum(ratio * advantages, clipped).mean()

        value_loss = ((values - trajectories.returns) ** 2).mean()
        entropy = -(softmax(logits) * log_softmax(logits)).sum(-1).mean()

        return policy_loss + 0.5 * value_loss - 0.01 * entropy

    grads = jax.grad(loss_fn)(policy.params)
    return grads
```

## Hidden State Management in PSRO

```python
def train_br_recurrent(opponent_pop, meta_strategy, num_episodes):
    policy = RecurrentPolicy(obs_dim, action_dim, hidden_dim=128)

    for episode in range(num_episodes):
        opp_idx = sample(meta_strategy)
        opponent = opponent_pop[opp_idx]

        carry_learner = policy.init_carry()
        carry_opponent = opponent.init_carry()

        obs = env.reset()
        done = False
        while not done:
            logits, _, carry_learner = policy(obs, carry_learner)
            action = sample_action(logits)

            opp_logits, _, carry_opponent = opponent(obs, carry_opponent)
            opp_action = sample_action(opp_logits)

            obs, reward, done = env.step(action, opp_action)
```

## References

- [PureJaxRL](https://github.com/luchris429/purejaxrl) — Fully JIT'd RL in JAX
- [Flax NNX RNN Guide](https://flax.readthedocs.io/en/latest/nnx/guides/transforms.html)

## Related

- [[RNN_Policies]] — Quick overview
- [[RL_Methods_for_PSRO]] — RL algorithms
- [[JAX_Tools]] / [[JAX_JIT_Practices]] — JAX ecosystem & patterns
