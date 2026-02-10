# NeuPL — Detailed Implementation Notes

> **Quick overview**: [[NeuPL]]

## Paper

**Title**: NeuPL: Neural Population Learning

**Authors**: Siqi Liu, Luke Marris, Daniel Hennes, Josh Merel, Nicolas Heess, Thore Graepel (DeepMind)

**Year**: 2022

**ArXiv**: [2202.07415](https://arxiv.org/abs/2202.07415)

**Published**: ICLR 2022

## Motivation

PSRO maintains **separate networks** per population policy → O(N) memory and no transfer learning between policies. NeuPL compresses the entire population into a **single conditional network**, giving:
- O(1) memory
- Automatic skill transfer via shared representations
- Faster convergence through joint training

## Core Algorithm

```
Initialize: Conditional network Π_θ(a|s,i) with 1 policy index
Initialize: Meta-strategy σ over population

for iteration t = 1 to convergence:
    # Phase 1: Train existing policies concurrently
    for each policy index i in population:
        Sample opponent j ~ σ
        Collect trajectories: Π_θ(·|·,i) vs Π_θ(·|·,j)
        Update θ to maximize u(i, j)

    # Phase 2: Add new best-response
    new_index = |population| + 1
    Train Π_θ(·|·,new_index) as BR to σ

    # Phase 3: Update meta-game
    Evaluate payoff matrix M[i,j] = u(Π_θ(·|·,i), Π_θ(·|·,j))
    σ = NashSolver(M)
```

### Key Differences from PSRO

| Aspect | PSRO | NeuPL |
|--------|------|-------|
| **Networks** | N separate | 1 conditional |
| **Memory** | O(N) | O(1) |
| **Transfer** | None | Via shared trunk |
| **Training** | Sequential BR | Concurrent + new BR |
| **Forgetting** | Not applicable | Must mitigate |

## Network Architecture

### Conditional Policy Network

```
Input: (observation s, policy_index i)
    ↓
Observation Encoder (shared across policies)
    - Conv layers (spatial) or MLP (vector)
    → obs_features
    ↓
Policy Index Encoding
    Option A: One-hot(i) → Linear → embedding
    Option B: Learned embedding table
    → policy_embedding ∈ ℝᵈ
    ↓
Conditioning Mechanism
    Option 1 — Concatenation:
        [obs_features, policy_embedding] → MLP → logits
    Option 2 — FiLM (Feature-wise Linear Modulation):
        γ, β = MLP(policy_embedding)
        features = γ * obs_features + β → MLP → logits
    Option 3 — Attention:
        Attention(obs_features, policy_embedding) → MLP → logits
    ↓
Policy Head → Softmax(logits) = π(a|s,i)
Value Head → v(s,i) ∈ ℝ
```

### FiLM Conditioning (Recommended)

```python
class FiLMLayer(nnx.Module):
    def __init__(self, feature_dim, cond_dim):
        self.gamma_proj = nnx.Linear(cond_dim, feature_dim)
        self.beta_proj = nnx.Linear(cond_dim, feature_dim)

    def __call__(self, features, condition):
        gamma = self.gamma_proj(condition)
        beta = self.beta_proj(condition)
        return gamma * features + beta
```

## Training Details

### Concurrent Training

All policies in population train simultaneously every iteration:

```python
def train_step(θ, population_size, meta_strategy, key):
    total_loss = 0
    for i in range(population_size):
        # Sample opponent from meta-strategy
        key, subkey = jax.random.split(key)
        j = jax.random.categorical(subkey, jnp.log(meta_strategy))

        # Self-play
        trajectories = play_game(π_θ(·|·,i), π_θ(·|·,j))

        # RL loss for policy i
        loss_i = compute_rl_loss(trajectories, policy_index=i)
        total_loss += loss_i

    # Single gradient step for all policies
    grads = jax.grad(total_loss)(θ)
    θ = optimizer.update(grads, θ)
    return θ
```

### Transfer Learning

New policies benefit from representations learned by earlier ones:
- **Shared encoder** already knows game dynamics
- **New BR head** only needs to specialize strategy
- **Faster convergence** at each iteration (empirically 2-5x)

### Catastrophic Forgetting Prevention

When training new policies, existing policies can degrade. Mitigations:
1. **Regularization**: KL penalty between current and reference policy
2. **Replay**: Periodically re-train existing policies
3. **Concurrent training**: Train all policies each iteration

## Meta-Strategy Computation

NeuPL uses the same meta-solvers as PSRO (Nash, PRD, α-Rank).

**Efficient Evaluation**: Because all policies share one network, evaluating the payoff matrix only requires changing the conditioning index — no model loading.

## Convergence

Under ideal conditions (exact BR, no forgetting), NeuPL converges to approximate Nash equilibrium. In practice:
- Approximate BR introduces bounded error
- Concurrent training helps convergence
- KL regularization controls forgetting

## Hyperparameters

| Parameter | Value Range |
|-----------|-------------|
| Policy embedding dim | 32–128 |
| Conditioning | FiLM or Concatenation |
| Concurrent training steps | 100–1000 per iter |
| BR training steps | 1000–10000 |
| KL regularization weight | 0.01–1.0 |
| Population growth rate | 1 per iteration |

## Evaluation Metrics

1. **Exploitability**: Same as PSRO
2. **NashConv**: Sum of per-player exploitability
3. **Policy Diversity**: Behavioral variance, KL between policy pairs
4. **Transfer Efficiency**: Steps to converge new BR vs from scratch

## Benchmark Tasks

- Running-with-Scissors (non-transitive)
- MuJoCo competitive tasks
- Poker variants
- OpenSpiel games

## Advantages

1. **Memory**: O(1) vs O(N) for PSRO
2. **Transfer**: Shared representations accelerate learning
3. **Efficiency**: Evaluate payoff matrix without model loading
4. **Scalability**: Handles large populations gracefully
5. **Continuous improvement**: All policies maintain competency

## Limitations

1. **Catastrophic forgetting**: Must actively mitigate
2. **Network capacity**: Single network must represent diverse strategies
3. **Conditioning quality**: Poor conditioning → poor diversity
4. **Symmetric zero-sum**: Original version limited to this setting (see [[NeuPL_JPSRO]] for general-sum)

## Extensions

- **[[Simplex_NeuPL]]**: Any-mixture Bayes-optimality
- **[[NeuPL_JPSRO]]**: General-sum, n-player, CCE convergence

## Code Resources

- [OpenSpiel](https://github.com/google-deepmind/open_spiel) (includes PSRO/NeuPL variants)
- [NeuPL Paper](https://arxiv.org/abs/2202.07415)

## References

- [NeuPL: Neural Population Learning (Liu et al., ICLR 2022)](https://arxiv.org/abs/2202.07415)
- [A Unified Game-Theoretic Approach to MARL (Lanctot et al., 2017)](https://arxiv.org/abs/1711.00832)

## Related

- [[NeuPL]] — Quick overview
- [[PSRO]] / [[PSRO_detailed]] — Population-based predecessor
- [[Simplex_NeuPL]] — Mixture-optimal extension
- [[NeuPL_JPSRO]] — General-sum extension
- [[RNN_Policies]] — Recurrent implementations in JAX
