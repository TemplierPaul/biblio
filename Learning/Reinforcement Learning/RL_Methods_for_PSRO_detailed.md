# RL Methods for PSRO — Detailed Implementation Notes

> **Quick overview**: [[RL_Methods_for_PSRO]]

## MPO (Maximum a Posteriori Policy Optimization)

### Algorithm

```python
for iteration in range(max_iterations):
    D = collect_trajectories(π_θ)
    A(s,a) = Q(s,a) - V(s)  # Advantages

    # E-step: Non-parametric target
    weights[a] = exp(A(s,a) / η)
    q(a|s) = weights / sum(weights)

    # M-step: KL-constrained fit
    loss = KL(q(·|s) || π_θ(·|s)) + λ·KL(π_θ || π_old)
    θ ← θ - α·∇loss
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| η (temperature) | 0.01–1.0 |
| ε_mean (KL bound) | 0.001–0.01 |
| ε_cov (KL bound) | 0.0001–0.001 |
| λ (regularization) | 0.1–1.0 |
| Batch size | 256–2048 |

### Code
- [DeepMind Acme MPO](https://github.com/google-deepmind/acme/tree/master/acme/agents/jax/mpo)
- [V-MPO](https://github.com/Matt00n/PolicyGradientsJax)

---

## CrossQ

### Algorithm

```python
for step in range(max_steps):
    a ~ π_θ(·|s)
    s', r = env.step(a)
    buffer.add((s, a, r, s'))

    batch = buffer.sample()

    # Critic (with BatchNorm, NO target network)
    Q_target = r + γ·Q_φ(s', π_θ(s'))
    loss_Q = (Q_φ(s, a) - Q_target)²

    # Actor
    loss_π = -Q_φ(s, π_θ(s))
```

### Key Differences from TD3/SAC

| Feature | TD3/SAC | CrossQ |
|---------|---------|--------|
| Target networks | ✓ | ✗ |
| BatchNorm in critic | ✗ | ✓ |
| UTD ratio | 1–20 | 1 |
| Complexity | Higher | Lower |

### Code
- [CrossQ Official](https://github.com/adityab/CrossQ)
- [SBX (Stable Baselines JAX)](https://github.com/araffin/sbx)

---

## Conditional Policies (NeuPL-style)

```python
class ConditionalPolicy(nnx.Module):
    def __init__(self, state_dim, action_dim, num_policies, embed_dim=64):
        self.policy_embed = nnx.Embed(num_policies, embed_dim)
        self.trunk = nnx.Sequential([
            nnx.Linear(state_dim + embed_dim, 256), nnx.relu,
            nnx.Linear(256, 256), nnx.relu,
        ])
        self.action_head = nnx.Linear(256, action_dim)

    def __call__(self, state, policy_idx):
        emb = self.policy_embed(policy_idx)
        x = jnp.concatenate([state, emb], axis=-1)
        return self.action_head(self.trunk(x))
```

## Distributed BR Training

```python
@jax.vmap
def parallel_br_training(seeds, opponent_params, config):
    def single_worker(seed, opp_params):
        key = jax.random.PRNGKey(seed)
        agent = initialize_agent(key, config)
        for step in range(config.training_steps):
            batch = collect_batch(agent.policy, opp_params, key)
            agent.update(batch)
        return agent.policy.params
    return single_worker(seeds, opponent_params)
```

## References

- [MPO (Abdolmaleki et al., 2018)](https://arxiv.org/abs/1806.06920)
- [V-MPO (Song et al., 2019)](https://arxiv.org/abs/1909.12238)
- [CrossQ (Bhatt et al., ICLR 2024)](https://openreview.net/forum?id=PczQtTsTIX)
- [SAC (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)

## Related

- [[RL_Methods_for_PSRO]] — Quick overview
- [[PSRO]] / [[NeuPL]] — Frameworks using these methods
- [[RNN_Policies]] — Recurrent implementations
- [[JAX_Tools]] — JAX ecosystem
