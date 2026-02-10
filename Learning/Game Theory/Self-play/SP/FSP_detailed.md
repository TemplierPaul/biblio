# Fictitious Self-Play — Detailed Implementation Notes

> **Quick overview**: [[FSP]]

## Paper

**Title**: Deep Reinforcement Learning from Self-Play in Imperfect-Information Games

**Authors**: Johannes Heinrich, David Silver (DeepMind)

**Year**: 2016

**ArXiv**: [1603.01121](https://arxiv.org/abs/1603.01121)

## Classical Fictitious Play (Brown, 1951)

```
Initialize: Each player chooses an arbitrary action
for iteration t = 1 to convergence:
    for each player i:
        Compute empirical frequency of opponent actions
        aᵢ(t) = best_response(opponent_empirical_distribution)
        Add aᵢ(t) to player i's action history
```

**Properties**: Converges to Nash in two-player zero-sum games.

## Neural Fictitious Self-Play (NFSP)

### Key Innovation

NFSP is the first scalable end-to-end approach to learning approximate Nash equilibria in large imperfect-information games. It combines:
1. **RL network**: Learn approximate best-response (DQN-style)
2. **SL network**: Learn average strategy (behavioral cloning)
3. **Self-play**: Train against copies of itself

### Two Networks Architecture

#### Best-Response Network Q(s,a|θ_Q)
- **Role**: Approximate best response to opponent's strategy
- **Training**: DQN-style RL
- **Policy**: ε-greedy: β = ε-greedy(Q)

#### Average Strategy Network Π(s,a|θ_Π)
- **Role**: Model agent's own average historical strategy
- **Training**: Supervised learning (behavioral cloning)
- **Policy**: Direct sampling from Π

### Two Memory Buffers

| Buffer | Type | Stores | Purpose |
|--------|------|--------|---------|
| **M_RL** | Circular/FIFO | [s, a, r, s'] | Train Q network |
| **M_SL** | Reservoir | [s, a] | Train Π network |

### Training Loop

```python
class NFSP:
    def __init__(self, state_dim, n_actions):
        self.Q = QNetwork(state_dim, n_actions)
        self.Pi = PolicyNetwork(state_dim, n_actions)
        self.M_RL = CircularBuffer(size=2_000_000)
        self.M_SL = ReservoirBuffer(size=2_000_000)
        self.eta = 0.1   # Anticipatory parameter
        self.epsilon = 0.06

    def select_action(self, state):
        if random.random() < self.eta:
            # Best-response mode
            if random.random() < self.epsilon:
                return random.choice(actions)
            else:
                return argmax(self.Q(state))
        else:
            # Average strategy mode
            return sample(self.Pi(state))

    def train_step(self):
        # Update Q-network (DQN loss)
        batch = self.M_RL.sample(batch_size)
        loss_Q = dqn_loss(batch, self.Q, self.Q_target)
        optimize(self.Q, loss_Q)

        # Update Π-network (cross-entropy loss)
        batch = self.M_SL.sample(batch_size)
        loss_Pi = cross_entropy_loss(batch, self.Pi)
        optimize(self.Pi, loss_Pi)
```

### Behavior Policy

```
σ(a|s) = {
    ε-greedy(Q(s,·))  with probability η     (best-response)
    Π(s,·)            with probability (1-η)  (average strategy)
}
```

### Reservoir Sampling (M_SL)

```python
def reservoir_add(buffer, item):
    if len(buffer) < capacity:
        buffer.append(item)
    else:
        idx = random.randint(0, total_items_seen)
        if idx < capacity:
            buffer[idx] = item
    total_items_seen += 1
```

## Network Architectures

```
Q-Network:                          Π-Network:
  Input → Linear(dim, 128)           Input → Linear(dim, 128)
  → ReLU → Linear(128, 128)          → ReLU → Linear(128, 128)
  → ReLU → Linear(128, n_actions)    → ReLU → Linear(128, n_actions)
  Output: Q-values                    → Softmax
                                      Output: Action probabilities
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| M_RL size | 200k – 2M |
| M_SL size | 200k – 2M |
| LR (Q) | 0.01 – 0.001 |
| LR (Π) | 0.01 – 0.001 |
| Batch size | 128 – 256 |
| η (anticipatory) | 0.1 |
| ε (exploration) | 0.06 – 0.1 |
| γ (discount) | 0.99 – 1.0 |
| Target network update | 100 – 1000 steps |

## Evaluation Metrics

1. **Exploitability**: Distance from Nash equilibrium
2. **Average Strategy Exploitability**: Π network alone
3. **Best Response Value**: Q network performance
4. **Nash Distance**: When exact Nash is known

## Variants

| Variant | Innovation |
|---------|-----------|
| **RM-FSP** (2023) | Regret minimization for BR |
| **MC-NFSP** (2019) | Monte Carlo for extensive-form |
| **NFSP for Many Players** | Extension to >2 players |
| **DiffFP** (2025) | Diffusion-based policy representations |

## Comparison with Other Methods

| Method | Info Type | Equilibrium | Scalability |
|--------|-----------|-------------|-------------|
| **NFSP** | Imperfect | Nash approx | High |
| **CFR** | Imperfect | Nash exact | Medium |
| **AlphaZero** | Perfect | Optimal | High |
| **PSRO** | Both | Nash approx | Medium |

## Code Resources

- [OpenSpiel NFSP](https://github.com/google-deepmind/open_spiel)
- [GitHub Implementation](https://github.com/dantodor/Neural-Ficititious-Self-Play-in-Imperfect-Information-Games)

## References

- [Deep RL from Self-Play in Imperfect-Info Games (Heinrich & Silver, 2016)](https://arxiv.org/abs/1603.01121)
- [RM-FSP (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0925231223005945)
- [Monte Carlo NFSP (2019)](https://arxiv.org/abs/1903.09569)

## Related

- [[FSP]] — Quick overview
- [[PSRO]] / [[PSRO_detailed]] — Population-based alternative
- [[CFR]] / [[CFR_detailed]] — Regret-minimization alternative
- [[AlphaZero]] / [[AlphaZero_detailed]] — Perfect-info self-play
